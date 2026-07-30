"""The consolidated pose encoder must reproduce the one it replaces.

``PoseEncoder`` was lifted out of an eval script that three corpus builders
imported from and a benchmark had copied. Those call sites produce published
numbers, so the ligand-side encoding has to be unchanged: the reference below is
the pre-consolidation ``_PoseEncoder`` body, kept here so the equivalence is
asserted rather than assumed.

The pocket side needs a real receptor and is covered end to end by the
benchmarks; what is checked here is the token assembly and the batching, which
is where a refactor of this shape breaks.
"""

from __future__ import annotations

import numpy as np
import torch

from prolit.data.descriptors import collate_molecules
from prolit.tokenizers.lm_vocab import AtomLMVocab
from prolit.tokenizers.pose_encoder import PoseEncoder

_DIM = 33
_CODEBOOK = 64


class _FakeTokenizer:
    """Deterministic stand-in: code = |round(row sum)| mod codebook."""

    def encode_batch(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        codes = x.sum(dim=-1).round().long().abs() % _CODEBOOK
        return codes.masked_fill(~mask, -1)


class _FixedDescriptors:
    """Stands in for LigandAtomDescriptor: returns a canned array per mol."""

    def compute(self, atoms, bonds, frame) -> tuple:  # noqa: ANN001, ARG002
        return atoms, None, None


def _reference_ligand_seqs_batch(  # noqa: PLR0913
    tokenizer: _FakeTokenizer,
    vocab: AtomLMVocab,
    mean: np.ndarray,
    std: np.ndarray,
    protein_codes: list[int],
    descs: list[np.ndarray],
) -> list[list[int] | None]:
    """The pre-consolidation ``_PoseEncoder.ligand_seqs_batch`` body."""
    valid = [(i, d) for i, d in enumerate(descs) if d.shape[0] > 0]
    out: list[list[int] | None] = [None] * len(descs)
    if not valid:
        return out
    tensors = [torch.from_numpy((d - mean) / std).float() for _, d in valid]
    x, mask = collate_molecules(tensors)
    idx = tokenizer.encode_batch(x, mask)
    for k, (i, _) in enumerate(valid):
        out[i] = vocab.build_sequence(protein_codes, idx[k][mask[k]].tolist())
    return out


def _encoder(mean: np.ndarray, std: np.ndarray) -> PoseEncoder:
    enc = PoseEncoder(
        _FakeTokenizer(),
        mean,
        std,
        AtomLMVocab(codebook_size=_CODEBOOK),
        torch.device("cpu"),
        pocket_cfg=None,
    )
    enc.lig_desc = _FixedDescriptors()
    return enc


def _stats() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return rng.normal(size=_DIM), rng.uniform(0.5, 2.0, size=_DIM)


def _mols(rng: np.random.Generator, lengths: list[int]) -> list[dict]:
    return [
        {"atoms": rng.normal(size=(n, _DIM)), "bonds": None} if n else
        {"atoms": np.zeros((0, _DIM)), "bonds": None}
        for n in lengths
    ]


def test_batch_matches_reference() -> None:
    rng = np.random.default_rng(3)
    mean, std = _stats()
    mols = _mols(rng, [7, 3, 11, 5])
    enc = _encoder(mean, std)
    protein = [1, 2, 3]

    got = enc.ligand_seqs_batch(protein, mols, frame=None)
    want = _reference_ligand_seqs_batch(
        _FakeTokenizer(), enc.vocab, mean, std, protein, [m["atoms"] for m in mols]
    )
    assert got == want


def test_empty_poses_come_back_as_none_in_place() -> None:
    """A pose with no encodable atoms must not shift its neighbours."""
    rng = np.random.default_rng(5)
    mean, std = _stats()
    mols = _mols(rng, [4, 0, 6])
    enc = _encoder(mean, std)

    got = enc.ligand_seqs_batch([9], mols, frame=None)
    assert got[1] is None
    assert got[0] is not None
    assert got[2] is not None

    solo = _encoder(mean, std).ligand_seqs_batch([9], [mols[0]], frame=None)
    assert got[0] == solo[0]


def test_all_empty_returns_all_none() -> None:
    mean, std = _stats()
    enc = _encoder(mean, std)
    assert enc.ligand_seqs_batch([1], _mols(np.random.default_rng(7), [0, 0]),
                                 frame=None) == [None, None]


def test_seqs_from_descs_chunking_is_transparent() -> None:
    """The chunked path must not depend on where the chunk boundaries fall."""
    rng = np.random.default_rng(11)
    mean, std = _stats()
    descs = [rng.normal(size=(n, _DIM)) for n in (5, 5, 5, 5, 5)]
    enc = _encoder(mean, std)
    runs = [
        enc.seqs_from_descs([2], descs, batch_size=b) for b in (1, 2, 64)
    ]
    assert runs[0] == runs[1] == runs[2]


def test_pocket_rotation_requires_setup() -> None:
    """Rotating before setup_pocket is a programming error, not a silent None."""
    enc = _encoder(*_stats())
    try:
        enc.pocket_codes_rotated(np.eye(3))
    except RuntimeError:
        return
    msg = "expected RuntimeError before setup_pocket()"
    raise AssertionError(msg)

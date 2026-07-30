"""The shared token-stream encoder must reproduce the per-script ones exactly.

Four corpus builders each carried their own copy of this flush loop; two were
byte-identical. Consolidating them is only safe if the emitted token stream is
unchanged, because every trained checkpoint is tied to the exact byte layout of
the caches these builders write. The reference implementations below are the
pre-consolidation code, kept here so the equivalence is asserted rather than
assumed.
"""

from __future__ import annotations

import numpy as np
import torch

from prolit.data.descriptors import collate_molecules
from prolit.data.token_stream import ComplexTokenEncoder
from prolit.tokenizers.lm_vocab import AtomLMVocab

_DIM = 33
_CODEBOOK = 64


class _FakeTokenizer:
    """Deterministic stand-in: code = round(sum of the row) mod codebook."""

    def encode_batch(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        codes = x.sum(dim=-1).round().long().abs() % _CODEBOOK
        return codes.masked_fill(~mask, -1)


class _CollectingWriter:
    """SplitWriter stand-in that keeps the sequences instead of writing them."""

    def __init__(self) -> None:
        self.docs: list[list[int]] = []

    def write(self, seqs: list[list[int]]) -> None:
        self.docs.extend(seqs)


def _reference_complex(
    tokenizer: _FakeTokenizer,
    vocab: AtomLMVocab,
    mean: np.ndarray,
    std: np.ndarray,
    rows: list[tuple[np.ndarray, np.ndarray | None]],
) -> list[list[int]]:
    """The pre-consolidation ``_Encoder.flush`` from the BioLIP/PLINDER builders."""

    def norm(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy((arr - mean) / std).float()

    def encode(descs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x, mask = collate_molecules(descs)
        return tokenizer.encode_batch(x, mask), mask

    prot = [norm(p) for p, _ in rows]
    ligs = [norm(lig) if lig is not None else None for _, lig in rows]
    pidx, pmask = encode(prot)
    if all(x is None for x in ligs):
        return [
            vocab.build_sequence(pidx[i][pmask[i]].tolist(), [])
            for i in range(len(prot))
        ]
    lidx, lmask = encode([x for x in ligs if x is not None])
    return [
        vocab.build_sequence(pidx[i][pmask[i]].tolist(), lidx[i][lmask[i]].tolist())
        for i in range(len(prot))
    ]


def _reference_ligand_only(
    tokenizer: _FakeTokenizer,
    vocab: AtomLMVocab,
    mean: np.ndarray,
    std: np.ndarray,
    rows: list[np.ndarray],
) -> list[list[int]]:
    """The pre-consolidation ``_Tokenizer.flush`` from the GEOM builder."""
    buf = [torch.from_numpy((r - mean) / std).float() for r in rows]
    x, mask = collate_molecules(buf)
    idx = tokenizer.encode_batch(x, mask)
    return [
        vocab.build_sequence([], idx[i][mask[i]].tolist()) for i in range(len(buf))
    ]


def _stats() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return rng.normal(size=_DIM), rng.uniform(0.5, 2.0, size=_DIM)


def _rows(rng: np.random.Generator, lengths: list[int]) -> list[np.ndarray]:
    return [rng.normal(size=(n, _DIM)) for n in lengths]


def test_matches_reference_on_complexes() -> None:
    rng = np.random.default_rng(7)
    mean, std = _stats()
    vocab = AtomLMVocab(codebook_size=_CODEBOOK)
    proteins = _rows(rng, [12, 5, 9])
    ligands = _rows(rng, [4, 7, 3])
    rows = list(zip(proteins, ligands, strict=True))

    writer = _CollectingWriter()
    enc = ComplexTokenEncoder(
        _FakeTokenizer(), vocab, mean, std, {"train": writer}, 16, torch.device("cpu")
    )
    for p, lig in rows:
        enc.add("train", p, lig)
    enc.flush_all()

    assert writer.docs == _reference_complex(
        _FakeTokenizer(), vocab, mean, std, rows
    )


def test_matches_reference_on_protein_only() -> None:
    rng = np.random.default_rng(11)
    mean, std = _stats()
    vocab = AtomLMVocab(codebook_size=_CODEBOOK)
    proteins = _rows(rng, [8, 14])
    rows: list[tuple[np.ndarray, np.ndarray | None]] = [(p, None) for p in proteins]

    writer = _CollectingWriter()
    enc = ComplexTokenEncoder(
        _FakeTokenizer(), vocab, mean, std, {"train": writer}, 16, torch.device("cpu")
    )
    for p, _ in rows:
        enc.add("train", p, None)
    enc.flush_all()

    assert writer.docs == _reference_complex(
        _FakeTokenizer(), vocab, mean, std, rows
    )


def test_matches_reference_on_ligand_only() -> None:
    rng = np.random.default_rng(13)
    mean, std = _stats()
    vocab = AtomLMVocab(codebook_size=_CODEBOOK)
    ligands = _rows(rng, [6, 2, 11])

    writer = _CollectingWriter()
    enc = ComplexTokenEncoder(
        _FakeTokenizer(), vocab, mean, std, {"train": writer}, 16, torch.device("cpu")
    )
    for lig in ligands:
        enc.add_ligand("train", lig)
    enc.flush_all()

    assert writer.docs == _reference_ligand_only(
        _FakeTokenizer(), vocab, mean, std, ligands
    )


def test_batching_does_not_change_output() -> None:
    """Flushing every row separately must give the same stream as one big flush."""
    rng = np.random.default_rng(17)
    mean, std = _stats()
    vocab = AtomLMVocab(codebook_size=_CODEBOOK)
    proteins = _rows(rng, [7, 3, 10, 5])
    ligands = _rows(rng, [2, 6, 4, 8])

    docs = []
    for batch_size in (1, 2, 64):
        writer = _CollectingWriter()
        enc = ComplexTokenEncoder(
            _FakeTokenizer(),
            vocab,
            mean,
            std,
            {"train": writer},
            batch_size,
            torch.device("cpu"),
        )
        for p, lig in zip(proteins, ligands, strict=True):
            enc.add("train", p, lig)
        enc.flush_all()
        docs.append(writer.docs)

    assert docs[0] == docs[1] == docs[2]


def test_mixed_pocketed_and_pocketless_rows_stay_aligned() -> None:
    """A ligand-only row in a complex batch must not shift the other rows' codes."""
    rng = np.random.default_rng(19)
    mean, std = _stats()
    vocab = AtomLMVocab(codebook_size=_CODEBOOK)
    protein = rng.normal(size=(9, _DIM))
    lig_a, lig_b = rng.normal(size=(4, _DIM)), rng.normal(size=(6, _DIM))

    writer = _CollectingWriter()
    enc = ComplexTokenEncoder(
        _FakeTokenizer(), vocab, mean, std, {"train": writer}, 16, torch.device("cpu")
    )
    enc.add("train", protein, lig_a)
    enc.add_ligand("train", lig_b)
    enc.flush_all()

    solo = _CollectingWriter()
    enc2 = ComplexTokenEncoder(
        _FakeTokenizer(), vocab, mean, std, {"train": solo}, 16, torch.device("cpu")
    )
    enc2.add("train", protein, lig_a)
    enc2.flush_all()

    assert writer.docs[0] == solo.docs[0]

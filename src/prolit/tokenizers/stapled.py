"""The stapled baseline's token stream: ESM3 pocket codes + ConfSeq + a pose.

This is the representation a reviewer proposes instead of ProLIT: take the best
published protein-only structure tokenizer, take the best published ligand-only
one, and concatenate their outputs into one sequence a language model can read.

    <bos> <p> ESM3 codes, one per pocket residue </p>
          <l> 3 pose codes, then ConfSeq tokens </l> <eos>

The pose codes are not decoration. ESM3 structure tokens describe per-residue
local geometry and ConfSeq is SMILES plus discretized internal angles; both are
SE(3)-invariant, so the concatenation says what the pocket looks like and what
the molecule looks like and **nothing about how they are arranged**. Without a
placement channel the stream cannot express a binding pose at all, and a model
trained on it would be answering a different question. So the baseline is handed
the missing rigid transform explicitly, quantized to a stated budget by
:mod:`prolit.tokenizers.pose_budget`, and the budget is reported as part of its
rate. ProLIT spends zero bits there.

Read every number this produces as a rate argument. Three facts, measured on the
PoseBusters benchmark's own 428 ligands, set the scale:

===========================  ==================  ====================
                             ProLIT              ESM3 + ConfSeq
===========================  ==================  ====================
tokens per ligand heavy atom  1.00                2.88 (ConfSeq)
bits per ligand heavy atom    13.0                24.8
placement                     in every token      0, plus this channel
===========================  ==================  ====================

Two envs, one vocabulary. ESM3's weights pull in a fork of ``transformers`` that
must not be installed beside the CLM's, so the ESM3 half is precomputed into a
per-structure token cache by the reconstruction benchmark's interpreter and read
back here. Everything in this module is numpy/RDKit only and imports ConfSeq
lazily, so it loads in either environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from prolit.tokenizers.lm_vocab import (
    BOS_ID,
    EOS_ID,
    L_CLOSE_ID,
    L_OPEN_ID,
    NUM_SPECIAL,
    P_CLOSE_ID,
    P_OPEN_ID,
)
from prolit.tokenizers.pose_budget import (
    DEFAULT_TOKEN_BITS,
    SemanticPoseCodec,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "DEFAULT_POSE_BITS",
    "ESM3_CODEBOOK_SIZE",
    "ConfSeqEncoding",
    "ConfSeqVocab",
    "StapledVocab",
    "confseq_decode",
    "confseq_encode",
    "pocket_box",
    "pose_of",
]

#: ESM3's structure codebook. Chain-break/BOS/EOS ids live above it in ESM3's
#: own space; the stream here never carries them, so only the 4096 codes map in.
ESM3_CODEBOOK_SIZE = 4096

#: The placement budget the language-model arm is trained with: four tokens,
#: 51.93 bits, laid out by
#: :class:`~prolit.tokenizers.pose_budget.SemanticPoseCodec`. More than the
#: reconstruction sweep's 39, and deliberately so -- at 39 the quantizer's own
#: error (0.75 A in translation, or 16 deg in rotation, depending on the split)
#: sits inside the 0-1 A band that decides docking power, so the experiment
#: would be measuring the quantizer rather than the representation. At 52 bits
#: the placement error is ~0.06 A. The extra bits are charged to the baseline
#: in the rate column and stated in the paper.
DEFAULT_POSE_BITS = 52


@dataclass(frozen=True)
class ConfSeqVocab:
    """String-to-id map for ConfSeq's whitespace-separated tokens.

    ConfSeq emits SMILES symbols and angle tokens like ``<-117>``; the set is
    small (391 distinct over the 428 PoseBusters ligands) but data-dependent, so
    it is built once over a corpus and shipped beside the token stream. An
    unseen token at encode time is a real event -- a molecule the corpus never
    contained -- and is reported rather than silently mapped to a bucket, since
    dropping it would change the molecule.
    """

    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", {t: i for i, t in enumerate(self.tokens)})

    @property
    def size(self) -> int:
        return len(self.tokens)

    def ids(self, tokens: Sequence[str]) -> list[int] | None:
        """Token strings to ids, or ``None`` if any token is out of vocabulary."""
        index = self._index  # type: ignore[attr-defined]
        out = []
        for t in tokens:
            i = index.get(t)
            if i is None:
                return None
            out.append(i)
        return out

    def strings(self, ids: Sequence[int]) -> list[str]:
        return [self.tokens[i] for i in ids]

    @classmethod
    def from_counts(cls, counts: dict[str, int]) -> ConfSeqVocab:
        """Build from a corpus pass, ordered by frequency then lexically.

        A deterministic order matters: the ids are baked into every token stream
        and into a trained model's embedding table, so rebuilding the vocabulary
        from the same corpus has to reproduce the same map.
        """
        return cls(tuple(sorted(counts, key=lambda t: (-counts[t], t))))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"tokens": list(self.tokens)}, indent=0))

    @classmethod
    def load(cls, path: Path) -> ConfSeqVocab:
        return cls(tuple(json.loads(Path(path).read_text())["tokens"]))


@dataclass(frozen=True)
class StapledVocab:
    """Flat LM vocabulary over three disjoint code ranges.

    ::

        0 .. 6                              specials, as AtomLMVocab
        7 .. 7+4095                         ESM3 structure codes (pocket)
        4103 .. 4103+8191                   pose codes (4 hierarchical tokens)
        12295 .. 12295+V-1                  ConfSeq tokens

    The three ranges are disjoint on purpose. Sharing one range, as ProLIT's
    single codebook does, would be a claim about the representation -- that a
    pocket code and a ligand code are the same kind of symbol -- and the whole
    point of this baseline is that they are not.

    The four pose tokens reuse one 8192-wide range; which scale a token refers
    to is its position, exactly as it is for a numeral.
    """

    confseq: ConfSeqVocab
    pose: SemanticPoseCodec = field(default_factory=SemanticPoseCodec)
    token_bits: int = DEFAULT_TOKEN_BITS

    @property
    def esm3_offset(self) -> int:
        return NUM_SPECIAL

    @property
    def pose_offset(self) -> int:
        return NUM_SPECIAL + ESM3_CODEBOOK_SIZE

    @property
    def pose_range(self) -> int:
        return 2**self.token_bits

    @property
    def confseq_offset(self) -> int:
        return self.pose_offset + self.pose_range

    @property
    def n_pose_tokens(self) -> int:
        return self.pose.n_tokens

    @property
    def pose_bits(self) -> float:
        return self.pose.bits

    @property
    def vocab_size(self) -> int:
        return self.confseq_offset + self.confseq.size

    def build_sequence(
        self,
        esm3_codes: Sequence[int],
        confseq_ids: Sequence[int],
        pose: Sequence[int] | None,
    ) -> list[int]:
        """Assemble one document.

        ``pose=None`` writes no placement tokens: that is the ligand-only
        regime (a GEOM conformer has no pocket to be placed in) and the
        zero-budget arm. Either block may be empty, which is what lets
        ligand-only and pocket-only corpora share this format.
        """
        seq = [BOS_ID, P_OPEN_ID]
        seq.extend(self.esm3_offset + int(c) for c in esm3_codes)
        seq.append(P_CLOSE_ID)
        seq.append(L_OPEN_ID)
        if pose is not None:
            if len(pose) != self.n_pose_tokens:
                msg = f"expected {self.n_pose_tokens} pose tokens, got {len(pose)}"
                raise ValueError(msg)
            seq.extend(self.pose_offset + int(d) for d in pose)
        seq.extend(self.confseq_offset + int(i) for i in confseq_ids)
        seq.append(L_CLOSE_ID)
        seq.append(EOS_ID)
        return seq

    def split_sequence(
        self, tokens: Sequence[int]
    ) -> tuple[list[int], tuple[int, ...] | None, list[int]]:
        """Inverse of :meth:`build_sequence`: (esm3 codes, pose tokens, confseq).

        The pose tokens are the leading tokens of the ligand block, so a
        truncated or malformed generation yields ``pose=None`` rather than a
        silently wrong placement.
        """
        esm3: list[int] = []
        pose_digits: list[int] = []
        confseq: list[int] = []
        mode: str | None = None
        for tok in tokens:
            if tok == P_OPEN_ID:
                mode = "p"
            elif tok == L_OPEN_ID:
                mode = "l"
            elif tok in (P_CLOSE_ID, L_CLOSE_ID):
                mode = None
            elif mode == "p" and self.esm3_offset <= tok < self.pose_offset:
                esm3.append(tok - self.esm3_offset)
            elif mode == "l" and self.pose_offset <= tok < self.confseq_offset:
                pose_digits.append(tok - self.pose_offset)
            elif mode == "l" and self.confseq_offset <= tok < self.vocab_size:
                confseq.append(tok - self.confseq_offset)
        pose = tuple(pose_digits) if len(pose_digits) == self.n_pose_tokens else None
        return esm3, pose, confseq


def pocket_box(backbone: np.ndarray) -> tuple[np.ndarray, float]:
    """The cube a pose is quantized inside, from the pocket alone.

    ``backbone`` is (N, 3) of the pocket's N/CA/C atoms -- the atoms ESM3
    reconstructs, so a receiver holding only the pocket block can compute the
    identical box. Deriving it from anything the ligand touches would leak the
    answer into the grid.
    """
    origin = backbone.min(axis=0)
    return origin, float((backbone.max(axis=0) - origin).max())


def pose_of(
    ligand: np.ndarray,
    reference: np.ndarray,
    backbone: np.ndarray,
    codec: SemanticPoseCodec | None = None,
) -> tuple[int, ...]:
    """Quantized placement taking ``reference`` (own frame) onto ``ligand``.

    ``reference`` is the molecule as the ligand-only tokenizer will decode it --
    its own canonical frame, no placement -- and ``ligand`` is where it actually
    sits. Rows must correspond.
    """
    codec = codec or SemanticPoseCodec()
    rot = _kabsch_rotation(reference, ligand)
    origin, size = pocket_box(backbone)
    return codec.encode(ligand.mean(axis=0), rot, origin, size)


def place(
    reference: np.ndarray,
    pose: Sequence[int] | None,
    backbone: np.ndarray,
    codec: SemanticPoseCodec | None = None,
) -> np.ndarray:
    """Apply a quantized placement to a decoded, own-frame molecule."""
    codec = codec or SemanticPoseCodec()
    origin, size = pocket_box(backbone)
    if pose is None:
        # Nothing was transmitted, so nothing can be recovered: the molecule
        # goes to the middle of the box in its own orientation. This is what
        # "concatenate the two tokenizers and stop" actually produces.
        return reference - reference.mean(axis=0) + (origin + size / 2.0)
    centroid, rot = codec.decode(tuple(pose), origin, size)
    return (reference - reference.mean(axis=0)) @ rot.T + centroid


def _kabsch_rotation(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    mc, tc = mobile.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((mobile - mc).T @ (target - tc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


# ---------------------------------------------------------------------------
# ConfSeq round trip. Imported lazily: ConfSeq's demo module does
# ``from indigo import *`` at import time, which not every environment has.
# ---------------------------------------------------------------------------


def _confseq_module(repo: Path):  # noqa: ANN202
    import sys  # noqa: PLC0415

    demo = str(Path(repo) / "demo")
    if demo not in sys.path:
        sys.path.insert(0, demo)
    import ConfSeq  # type: ignore[import-not-found]  # noqa: PLC0415

    return ConfSeq


@dataclass(frozen=True)
class ConfSeqEncoding:
    """One ligand through ConfSeq, with both frames in the SAME atom order.

    ``own`` is where the decoder puts the molecule (its own canonical frame,
    which is all the tokens carry) and ``reference`` is where the molecule
    actually sits. They are aligned row for row by the decoded molecule's
    substructure match onto the input, so the rigid transform between them is
    exactly the placement the tokens omit.
    """

    tokens: list[str]
    own: np.ndarray
    reference: np.ndarray


def confseq_encode(
    mol: Any,  # noqa: ANN401 -- an RDKit Mol; rdkit ships no usable stubs
    repo: Path,
) -> ConfSeqEncoding | None:
    """Encode an RDKit mol to ConfSeq tokens and decode them back.

    Returns ``None`` when the round trip fails -- a molecule ConfSeq cannot
    describe is a molecule this baseline cannot represent, and counting it as a
    success with the input coordinates would credit the representation with
    information it does not carry.
    """
    from rdkit import Chem  # noqa: PLC0415

    cs = _confseq_module(repo)
    try:
        _smiles, seq = cs.get_ConfSeq_pair_from_mol(cs.aug_mol(mol, 0))
        rec = cs.get_mol_from_ConfSeq_pair(
            cs.replace_angle_brackets_with_line(seq), seq
        )
        rec = Chem.MolFromMolBlock(
            cs.remove_degree_in_molblock(Chem.MolToMolBlock(rec))
        )
    except Exception:  # noqa: BLE001 -- third-party parser, many failure modes
        return None
    if rec is None or rec.GetNumConformers() == 0:
        return None
    match = mol.GetSubstructMatch(rec)
    if len(match) != rec.GetNumAtoms():
        return None
    return ConfSeqEncoding(
        tokens=seq.split(),
        own=np.asarray(rec.GetConformer().GetPositions(), dtype=np.float64),
        reference=np.asarray(
            mol.GetConformer().GetPositions()[list(match)], dtype=np.float64
        ),
    )


def confseq_decode(tokens: Sequence[str], repo: Path):  # noqa: ANN201
    """Decode ConfSeq tokens back to an RDKit mol in its own frame."""
    from rdkit import Chem  # noqa: PLC0415

    cs = _confseq_module(repo)
    seq = " ".join(tokens)
    try:
        rec = cs.get_mol_from_ConfSeq_pair(
            cs.replace_angle_brackets_with_line(seq), seq
        )
        rec = Chem.MolFromMolBlock(
            cs.remove_degree_in_molblock(Chem.MolToMolBlock(rec))
        )
    except Exception:  # noqa: BLE001
        return None
    if rec is None or rec.GetNumConformers() == 0:
        return None
    return rec

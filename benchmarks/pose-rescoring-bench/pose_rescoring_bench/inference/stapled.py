"""Scoring CASF poses with the ESM3 x ConfSeq baseline.

Presents the surface :mod:`pose_rescoring_bench.inference.rescoring` already
uses -- ``setup_pocket`` once per target, ``ligand_seq`` once per pose -- so the
stapled arm walks the identical evaluation: the same 285 targets, the same
docking decoys read from the same mol2 files, the same decoys-only protocol, the
same head architecture on top.

One asymmetry is worth stating rather than discovering in the numbers. The ESM3
pocket codes are **identical across every pose of a target** -- the receptor
does not move -- so everything that distinguishes one pose from another in this
stream lives in the four placement tokens and in ConfSeq's internal angles.
ProLIT's ligand block is ~25 atom tokens whose spherical coordinates are all
measured in the shared pocket frame, so its pose information is spread across
the whole block. That difference is the hypothesis under test, not a defect of
this adapter.

The pocket residues are ProLIT's own: the same extraction, the same
``max_residues``, so both arms condition on the same residues of the same
receptor and differ only in how those residues are written down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from prolit.tokenizers.stapled_encoder import StapledEncoder, StapledPocket

__all__ = ["StapledPoseEncoder", "make_stapled_encoder"]


@dataclass
class StapledPoseEncoder:
    """``PoseEncoder``-shaped façade over :class:`StapledEncoder`.

    ``needs_struct_id`` is how the caller knows to hand over the target id: the
    ESM3 tokens are cached per receptor, and the cache is keyed by the CASF
    directory name rather than by anything recoverable from the PDB text.
    """

    inner: StapledEncoder
    needs_struct_id: bool = True

    def setup_pocket(
        self,
        protein_text: str,
        reference_heavy: np.ndarray,
        struct_id: str,
    ) -> tuple[StapledPocket, None] | None:
        """Pocket codes for one target, or ``None`` if the cache misses it.

        Returns a ``(pocket, frame)`` pair to match ``PoseEncoder``; the frame
        is ``None`` because this arm has no shared frame -- which is the point.
        """
        pocket = self.inner.setup_pocket(struct_id, protein_text, reference_heavy)
        return None if pocket is None else (pocket, None)

    def ligand_seq(
        self,
        pocket: StapledPocket,
        mol: dict,
        _frame: Any = None,  # noqa: ANN401 -- unused; kept for the shared shape
    ) -> list[int] | None:
        """One pose as a stapled token stream."""
        heavy_idx = np.array(
            [i for i, a in enumerate(mol["atoms"]) if a[0] != "H"], dtype=int
        )
        if heavy_idx.size == 0:
            return None
        return self.inner.ligand_seq(pocket, mol["atoms"], mol["bonds"], heavy_idx)


def make_stapled_encoder(
    esm3_cache: Path,
    confseq_repo: Path,
    confseq_vocab: Path,
    max_residues: int,
) -> StapledPoseEncoder:
    """Build the façade from the cache and the frozen vocabulary.

    The vocabulary is the constant one every stapled corpus was written with.
    Passing a different one would renumber the stream under a model trained on
    the first, which trains and converges and means nothing.
    """
    # Imported here, not at module scope: ConfSeq needs Indigo, which only the
    # ``stapled`` dependency group installs, and importing this module must not
    # require it of a run that only scores the ProLIT arm.
    from prolit.config import PocketExtractionConfig  # noqa: PLC0415
    from prolit.data.esm3_tokens import Esm3TokenCache  # noqa: PLC0415
    from prolit.tokenizers.stapled import (  # noqa: PLC0415
        ConfSeqVocab,
        StapledVocab,
    )
    from prolit.tokenizers.stapled_encoder import StapledEncoder  # noqa: PLC0415

    return StapledPoseEncoder(
        inner=StapledEncoder(
            cache=Esm3TokenCache(esm3_cache),
            confseq_repo=confseq_repo,
            vocab=StapledVocab(confseq=ConfSeqVocab.load(confseq_vocab)),
            pocket_cfg=PocketExtractionConfig(max_residues=max_residues),
        )
    )

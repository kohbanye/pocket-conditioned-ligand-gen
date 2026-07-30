"""Shared data types passed between datasets, adapters, and the runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Sample:
    """One structure to reconstruct.

    A protein-only sample has ``ligand_sdf=None``; a full complex carries both.
    ``pocket_residue_index`` optionally restricts the protein to a residue
    subset (1-based author numbering) so every model sees the same region.
    """

    sample_id: str
    protein_pdb: Path | None = None
    ligand_sdf: Path | None = None
    chain: str | None = None
    pocket_residue_index: list[int] | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class ModalityRecon:
    """Aligned reference/reconstructed coordinates for one modality.

    ``ref`` and ``rec`` are (N, 3) and share row order (atom i in ``ref`` maps to
    atom i in ``rec``), so metrics can be computed without re-matching atoms.
    For ``protein_backbone`` the rows are CA atoms (one per residue) unless
    ``atom_kind`` says otherwise.
    """

    modality: str  # "protein_backbone" | "ligand"
    ref: np.ndarray  # (N, 3)
    rec: np.ndarray  # (N, 3)
    atom_kind: str = "CA"  # "CA" | "N,CA,C" | "heavy"
    n_residues: int | None = None
    n_tokens: int | None = None
    # Per-row residue identity (chain, author_resid), aligned with ref/rec rows.
    # Lets the runner restrict protein metrics to a residue subset (e.g. the
    # pocket) so a full-protein reconstruction can be scored on the same
    # residues as the pocket-only model.
    res_keys: list[tuple[str, int]] | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class ReconResult:
    """Everything one adapter produced for one sample."""

    model: str
    sample_id: str
    modalities: list[ModalityRecon] = field(default_factory=list)
    ok: bool = True
    error: str | None = None
    runtime_s: float | None = None
    extra: dict = field(default_factory=dict)

"""Shared data types passed between datasets, adapters, and the evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Target:
    """One docking target (a protein pocket) to generate ligands for.

    A "target" bundles everything the models and the evaluator need:

    * ``receptor_pdb``  — protein-only receptor (ATOM records, standard AAs).
    * ``pocket_pdb``    — the pocket residues only (≤10 Å of the reference
      ligand); some models (TargetDiff, DiffGui) condition on this directly.
    * ``ref_ligand_sdf``— the co-crystal / reference ligand. Defines the pocket,
      the docking box, and a positive-control pose, and is the reference for
      novelty / similarity / "better-than-reference" metrics.
    * ``receptor_pdbqt``— receptor prepared for AutoDock Vina.
    * ``box``           — ``{"center": [x,y,z], "size": [sx,sy,sz]}`` search box.
    """

    target_id: str
    receptor_pdb: Path
    ref_ligand_sdf: Path
    pocket_pdb: Path | None = None
    receptor_pdbqt: Path | None = None
    box: dict | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class GenResult:
    """What one adapter produced for one target.

    ``sdf`` holds every generated ligand as a 3D heavy-atom mol block (the source
    of truth handed to the evaluator). ``n_requested`` / ``n_generated`` record
    library size so comparisons can be made size-fair (a known confound: a model
    that emits 100× more molecules and keeps best-k looks artificially strong).
    """

    model: str
    target_id: str
    sdf: Path | None = None
    n_requested: int = 0
    n_generated: int = 0
    ok: bool = True
    error: str | None = None
    runtime_s: float | None = None
    extra: dict = field(default_factory=dict)

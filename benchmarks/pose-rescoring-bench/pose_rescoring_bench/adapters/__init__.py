"""Pose baselines, one adapter per published method under comparison.

Each adapter scores the CASF-2016 docking-power decoy stacks with the method's
own released weights, in the method's own environment, on the same poses and the
same RMSD reference our arms use. Published numbers are never copied into the
table: a row exists because it was measured here.

This replaces the older ``pose_rescoring_bench.baselines`` path, which parsed
per-target ``*_score.dat`` files that a sibling checkout had already produced.
That module stays, because the dumps it produced are what the current published
numbers were computed from, but it cannot be re-run anywhere the sibling
checkout does not exist. An adapter can.

``uses_explicit_contacts`` is carried on every adapter because it decides which
rows are a like-for-like comparison. Our tokenizer is handed no hand-specified
contacts, so the methods that are also given none are the closer peers.
"""

from __future__ import annotations

from pose_rescoring_bench.adapters.base import DUMP_COLUMNS, PoseBaseline

_REGISTRY = {
    # Interaction-featurised: a learned mixture density over protein-ligand
    # atom-pair distances, read off a residue-level pocket graph.
    "rtmscore": ("pose_rescoring_bench.adapters.rtmscore", "RTMScoreAdapter"),
    # Interaction-featurised: the same distance-likelihood formulation on an
    # atom-level pocket graph, retrained on a mixed docking/screening objective.
    "genscore": ("pose_rescoring_bench.adapters.genscore", "GenScoreAdapter"),
    # Interaction-featurised: the protein and the ligand are voxelised into one
    # joint grid, so every contact is directly visible to the CNN.
    "gnina": ("pose_rescoring_bench.adapters.gnina", "GninaAdapter"),
    # Empirical: a fitted functional form over atom-type pair terms, not a
    # trained network. The reference row rather than a peer.
    "vina": ("pose_rescoring_bench.adapters.vina", "VinaAdapter"),
}


def build(name: str, **kwargs: object) -> PoseBaseline:
    """Instantiate an adapter by name (imported lazily to avoid heavy deps)."""
    import importlib  # noqa: PLC0415

    if name not in _REGISTRY:
        msg = f"unknown pose baseline {name!r}; choose from {sorted(_REGISTRY)}"
        raise KeyError(msg)
    module_name, cls_name = _REGISTRY[name]
    return getattr(importlib.import_module(module_name), cls_name)(**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["DUMP_COLUMNS", "PoseBaseline", "available", "build"]

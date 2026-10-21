"""Affinity baselines, one adapter per published model under comparison.

Each adapter scores the CASF-2016 core set with the model's own released
weights, in the model's own environment, on the same poses and labels our arms
use. Published numbers are never copied into the table: a row exists because it
was measured here.

``uses_explicit_contacts`` is carried on every adapter because it decides which
rows are a like-for-like comparison. Our tokenizer is handed no hand-specified
contacts, so the methods that are also given none are the closer peers, and the
ones that featurise interactions directly measure something else.
"""

from __future__ import annotations

from affinity_bench.adapters.base import DUMP_COLUMNS, AffinityBaseline

_REGISTRY = {
    # Interaction-featurised: residue-atom contact shells.
    "onionnet2": ("affinity_bench.adapters.onionnet2", "OnionNet2Adapter"),
    # Interaction-featurised: every ligand-pocket atom pair within 5 A is an edge.
    "gign": ("affinity_bench.adapters.gign", "GIGNAdapter"),
    # Interaction-featurised: learned distance distributions.
    "genscore": ("affinity_bench.adapters.genscore", "GenScoreAdapter"),
    # NOT interaction-featurised: one joint voxel grid, 28 one-hot atom
    # types, no contacts. The first peer here that is handed none either.
    "sfcnn": ("affinity_bench.adapters.sfcnn", "SfcnnAdapter"),
    # NOT interaction-featurised, but pose-BLIND: pocket graph over CA
    # coordinates plus a 2D ligand graph. A reference row, not a peer.
    "planet": ("affinity_bench.adapters.planet", "PlanetAdapter"),
}


def build(name: str, **kwargs: object) -> AffinityBaseline:
    """Instantiate an adapter by name (imported lazily to avoid heavy deps)."""
    import importlib  # noqa: PLC0415

    if name not in _REGISTRY:
        msg = f"unknown affinity baseline {name!r}; choose from {sorted(_REGISTRY)}"
        raise KeyError(msg)
    module_name, cls_name = _REGISTRY[name]
    return getattr(importlib.import_module(module_name), cls_name)(**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["DUMP_COLUMNS", "AffinityBaseline", "available", "build"]

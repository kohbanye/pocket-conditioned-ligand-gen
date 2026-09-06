"""Reconstruction adapters, one per model under comparison."""

from __future__ import annotations

from recon_bench.adapters.base import ReconstructionModel

_REGISTRY = {
    "esm3": ("recon_bench.adapters.esm3", "ESM3Adapter"),
    "foldtoken": ("recon_bench.adapters.foldtoken", "FoldTokenAdapter"),
    # ProLIT. Instantiate with arm="joint" / "separate" / ...;
    # see recon_bench.adapters.own_allatom.ARMS.
    "own_allatom": ("recon_bench.adapters.own_allatom", "OwnAllAtomAdapter"),
    "token_mol": ("recon_bench.adapters.token_mol", "TokenMolAdapter"),
    "confseq": ("recon_bench.adapters.confseq", "ConfSeqAdapter"),
    "bio2token": ("recon_bench.adapters.bio2token", "Bio2TokenAdapter"),
    # ESM3 (pocket) + ConfSeq (ligand) concatenated, with the rigid transform
    # neither of them carries handed over as an explicit bit budget.
    # Instantiate with pose_bits=None|13|26|39; see the module docstring.
    "stapled": ("recon_bench.adapters.stapled", "StapledAdapter"),
}


def build(name: str, **kwargs) -> ReconstructionModel:
    """Instantiate an adapter by name (imported lazily to avoid heavy deps)."""
    import importlib

    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(_REGISTRY)}")
    module_name, cls_name = _REGISTRY[name]
    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls(**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["ReconstructionModel", "build", "available"]

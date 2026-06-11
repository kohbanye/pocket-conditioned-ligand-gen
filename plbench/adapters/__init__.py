"""Reconstruction adapters, one per model under comparison."""

from __future__ import annotations

from plbench.adapters.base import ReconstructionModel

_REGISTRY = {
    "esm3": ("plbench.adapters.esm3", "ESM3Adapter"),
    "foldtoken": ("plbench.adapters.foldtoken", "FoldTokenAdapter"),
    "own_vqvae": ("plbench.adapters.own_vqvae", "OwnVQVAEAdapter"),
    "token_mol": ("plbench.adapters.token_mol", "TokenMolAdapter"),
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

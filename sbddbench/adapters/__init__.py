"""Generation adapters, one per SBDD model under comparison."""

from __future__ import annotations

from sbddbench.adapters.base import GenerativeModel

_REGISTRY = {
    "own": ("sbddbench.adapters.own", "OwnAdapter"),
    "diffsbdd": ("sbddbench.adapters.diffsbdd", "DiffSBDDAdapter"),
    "targetdiff": ("sbddbench.adapters.targetdiff", "TargetDiffAdapter"),
    "diffgui": ("sbddbench.adapters.diffgui", "DiffGuiAdapter"),
}


def build(name: str, **kwargs) -> GenerativeModel:
    """Instantiate an adapter by name (imported lazily to avoid heavy deps)."""
    import importlib

    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(_REGISTRY)}")
    module_name, cls_name = _REGISTRY[name]
    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls(**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["GenerativeModel", "build", "available"]

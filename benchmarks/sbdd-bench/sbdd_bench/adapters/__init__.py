"""Generation adapters, one per SBDD model under comparison."""

from __future__ import annotations

from sbdd_bench.adapters.base import GenerativeModel

_REGISTRY = {
    "own": ("sbdd_bench.adapters.own", "OwnAdapter"),
    "diffsbdd": ("sbdd_bench.adapters.diffsbdd", "DiffSBDDAdapter"),
    "flowr": ("sbdd_bench.adapters.flowr", "FlowrAdapter"),
    "targetdiff": ("sbdd_bench.adapters.targetdiff", "TargetDiffAdapter"),
    "diffgui": ("sbdd_bench.adapters.diffgui", "DiffGuiAdapter"),
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

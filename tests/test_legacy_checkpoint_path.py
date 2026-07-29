"""The rename from ``src`` to ``prolit`` must not orphan trained checkpoints.

Lightning pickles a checkpoint's config dataclass into ``hyper_parameters``, and
pickle records the class by module path, so every checkpoint trained before the
rename references ``src.config.*``. If the compatibility finder regresses, those
checkpoints -- including every number the paper reports -- stop loading.
"""

from __future__ import annotations

import importlib
import pickle

import prolit  # noqa: F401  (importing it installs the finder)
from prolit.config import AtomVQVAETrainingConfig, LMTrainingConfig


def test_legacy_module_path_resolves() -> None:
    assert importlib.import_module("src.config") is importlib.import_module(
        "prolit.config"
    )


def test_legacy_pickle_of_a_config_loads() -> None:
    """A pickle naming the old module path unpickles to the current class."""
    original = AtomVQVAETrainingConfig.__module__
    try:
        # Reproduce what a pre-rename process would have written.
        AtomVQVAETrainingConfig.__module__ = "src.config"
        blob = pickle.dumps(AtomVQVAETrainingConfig(), protocol=2)
    finally:
        AtomVQVAETrainingConfig.__module__ = original

    assert b"src.config" in blob
    restored = pickle.loads(blob)  # noqa: S301
    assert isinstance(restored, AtomVQVAETrainingConfig)
    assert restored.atom.codebook_size == AtomVQVAETrainingConfig().atom.codebook_size


def test_nested_legacy_config_loads() -> None:
    """Training configs nest model configs; both module paths must resolve."""
    classes = [LMTrainingConfig, type(LMTrainingConfig().model)]
    originals = {cls: cls.__module__ for cls in classes}
    try:
        for cls in classes:
            cls.__module__ = "src.config"
        blob = pickle.dumps(LMTrainingConfig(), protocol=2)
    finally:
        for cls, mod in originals.items():
            cls.__module__ = mod

    restored = pickle.loads(blob)  # noqa: S301
    assert isinstance(restored, LMTrainingConfig)
    assert restored.model.vocab_size == LMTrainingConfig().model.vocab_size


def test_unknown_legacy_submodule_still_fails() -> None:
    """The alias only covers modules that exist, so typos surface as errors."""
    try:
        importlib.import_module("src.not_a_real_module")
    except ModuleNotFoundError:
        return
    msg = "expected src.not_a_real_module to be unimportable"
    raise AssertionError(msg)

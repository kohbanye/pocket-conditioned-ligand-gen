"""Renames must not orphan trained checkpoints.

Lightning pickles a checkpoint's config dataclass into ``hyper_parameters``, and
pickle records the class by *both* its module path and its class name. Two
renames have happened, and each needs its own compatibility shim:

* the package moved from ``src`` to ``prolit`` -- handled by the meta-path
  finder in :mod:`prolit._legacy_import_path`;
* the config classes took the paper's names (``LigandLMConfig`` ->
  ``ProLITCLMConfig`` and friends) -- handled by aliases at the bottom of
  :mod:`prolit.config`.

If either regresses, checkpoints stop loading -- including every number the
paper reports.
"""

from __future__ import annotations

import importlib
import pickle

import prolit  # noqa: F401  (importing it installs the finder)
from prolit import config
from prolit.config import AtomVQVAETrainingConfig, CLMTrainingConfig


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
    classes = [CLMTrainingConfig, type(CLMTrainingConfig().model)]
    originals = {cls: cls.__module__ for cls in classes}
    try:
        for cls in classes:
            cls.__module__ = "src.config"
        blob = pickle.dumps(CLMTrainingConfig(), protocol=2)
    finally:
        for cls, mod in originals.items():
            cls.__module__ = mod

    restored = pickle.loads(blob)  # noqa: S301
    assert isinstance(restored, CLMTrainingConfig)
    assert restored.model.vocab_size == CLMTrainingConfig().model.vocab_size


def test_unknown_legacy_submodule_still_fails() -> None:
    """The alias only covers modules that exist, so typos surface as errors."""
    try:
        importlib.import_module("src.not_a_real_module")
    except ModuleNotFoundError:
        return
    msg = "expected src.not_a_real_module to be unimportable"
    raise AssertionError(msg)


def test_configs_still_load_under_their_pre_rename_names() -> None:
    """Checkpoints name the config class they were pickled with, not today's."""
    for old, new in (
        ("LigandLMConfig", config.ProLITCLMConfig),
        ("LMTrainingConfig", config.CLMTrainingConfig),
        ("ComplexMLMConfig", config.ProLITMLMConfig),
    ):
        assert getattr(config, old) is new, old


def test_a_pickle_naming_the_old_config_class_loads() -> None:
    """The end-to-end path: an old name in the blob, the new class out of it."""
    original = CLMTrainingConfig.__module__, CLMTrainingConfig.__qualname__
    try:
        # Reproduce what a pre-rename process would have written.
        CLMTrainingConfig.__qualname__ = "LMTrainingConfig"
        blob = pickle.dumps(CLMTrainingConfig(), protocol=2)
    finally:
        CLMTrainingConfig.__module__, CLMTrainingConfig.__qualname__ = original

    assert b"LMTrainingConfig" in blob
    restored = pickle.loads(blob)  # noqa: S301
    assert isinstance(restored, CLMTrainingConfig)

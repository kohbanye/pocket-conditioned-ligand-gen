"""Let checkpoints written under the old ``src.*`` import path still load.

The package used to be importable as ``src``. Lightning pickles a checkpoint's
config **dataclass instance** into ``hyper_parameters``, and pickle stores the
class by its module path, so every checkpoint trained before the rename carries
``src.config.AtomVQVAETrainingConfig`` (and friends) inside it. Unpickling those
imports ``src.config``, which no longer exists.

Rather than rewrite ~165 run directories of published checkpoints -- including
the ones the paper's results are computed from -- this installs a meta-path
finder that resolves any ``src.*`` module to the matching ``prolit.*`` one. It
is registered from :mod:`prolit`, so it is in place before any loader in
:mod:`prolit.tokenizers.loaders` can call ``torch.load``.

``src`` itself resolves to an empty stub package, because pickle imports the
parent before the submodule and the real ``src/`` directory is only importable
as a namespace package when the process happens to run from the repository root.
The stub carries no attributes, so ``from src.config import X`` works while
``src.anything_else`` still fails.
"""

from __future__ import annotations

import importlib
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_OLD_ROOT = "src"
_NEW_ROOT = "prolit"


class _AliasLoader(Loader):
    """Return the ``prolit.*`` module under the requested ``src.*`` name."""

    def __init__(self, target: str) -> None:
        self._target = target

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        return importlib.import_module(self._target)

    def exec_module(self, module: ModuleType) -> None:
        """No-op: the aliased module was already executed under its real name."""


class _StubLoader(Loader):
    """Create an empty package to stand in for the old ``src`` root."""

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        module = ModuleType(spec.name)
        module.__path__ = []  # marks it a package so submodules may be imported
        return module

    def exec_module(self, module: ModuleType) -> None:
        """No-op: the stub holds nothing."""


class _LegacyPathFinder(MetaPathFinder):
    """Resolve ``src.<submodule>`` to ``prolit.<submodule>``."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,  # noqa: ARG002
        target: ModuleType | None = None,  # noqa: ARG002
    ) -> ModuleSpec | None:
        # Pickle imports the parent package first, so ``src`` must resolve even
        # though it holds nothing itself.
        if fullname == _OLD_ROOT:
            return ModuleSpec(fullname, _StubLoader(), is_package=True)
        if not fullname.startswith(_OLD_ROOT + "."):
            return None
        new_name = _NEW_ROOT + fullname[len(_OLD_ROOT) :]
        try:
            importlib.import_module(new_name)
        except ModuleNotFoundError:
            return None
        return ModuleSpec(fullname, _AliasLoader(new_name))


def install() -> None:
    """Register the finder once. Safe to call repeatedly."""
    if not any(isinstance(f, _LegacyPathFinder) for f in sys.meta_path):
        sys.meta_path.append(_LegacyPathFinder())

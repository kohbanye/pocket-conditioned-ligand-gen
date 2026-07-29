"""Inference drivers: load the sibling repo's trained models and produce dumps.

These modules import the source repo's *library* layer (``src.tokenizers.*``,
``src.model.*``, ``src.data.*``) — never its ``scripts/*`` eval layer — load a
variant's checkpoints, run each task, and write per-sample dumps in the
:mod:`ctbench.io_dumps` schema. They require torch + a GPU and the source repo on
the import path; they are meant to run under qsub, not in unit tests.

The source repo is intentionally NOT a build dependency (its setuptools config
scans huge data/wandb trees and hangs), so it is put on ``sys.path`` at import
time by :func:`ensure_source_repo_importable`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Sibling layout: <git>/complex-tokenizer-bench/ctbench/inference/__init__.py
# -> the source repo is <git>/pocket-conditioned-ligand-gen.
_GIT = Path("/gs/bs/tga-ohuelab/sakano/git")
_SIBLING = Path(__file__).resolve().parents[3] / "pocket-conditioned-ligand-gen"
_DEFAULT_SOURCE_REPO = _GIT / "pocket-conditioned-ligand-gen"


def ensure_source_repo_importable() -> None:
    """Put the source repo on ``sys.path`` if ``import src`` is not already resolvable.

    Honors ``CTBENCH_SOURCE_REPO``; otherwise tries the sibling directory and the
    known absolute location. A no-op once ``src`` is importable.
    """
    try:
        import src  # noqa: F401, PLC0415
    except ModuleNotFoundError:
        root = os.environ.get("CTBENCH_SOURCE_REPO")
        candidates = [Path(root)] if root else [_SIBLING, _DEFAULT_SOURCE_REPO]
        for candidate in candidates:
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
                return

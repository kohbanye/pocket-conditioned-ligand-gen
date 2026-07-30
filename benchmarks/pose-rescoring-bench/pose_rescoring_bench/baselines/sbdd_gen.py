"""SBDD generation baselines (DiffGui / TargetDiff / DiffSBDD): collect + rerun.

The generations and their metric computation live in the sbdd-bench repo. We
collect its official per-model / per-target summaries into this repo's
``results/generation/baselines/`` and expose a rerun wrapper.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

BASELINE_MODELS = ("diffgui", "targetdiff", "diffsbdd")


def collect_per_model(sbdd_bench_repo: Path) -> pd.DataFrame:
    """sbdd-bench ``results/per_model.csv`` (all models, one row each)."""
    return pd.read_csv(sbdd_bench_repo / "results" / "per_model.csv")


def collect_per_target(sbdd_bench_repo: Path) -> pd.DataFrame:
    """sbdd-bench ``results/per_target.csv`` (per model x target)."""
    return pd.read_csv(sbdd_bench_repo / "results" / "per_target.csv")


def rerun_evaluation(
    sbdd_bench_repo: Path,
    models: list[str],
    dock_modes: tuple[str, ...] = ("score", "min"),
    extra_args: list[str] | None = None,
) -> None:
    """Re-run the sbdd-bench evaluation for the given baseline models (subprocess)."""
    script = sbdd_bench_repo / "scripts" / "run_evaluation.py"
    cmd = [
        "python",
        str(script),
        "--models",
        *models,
        "--dock-modes",
        *dock_modes,
        *(extra_args or []),
    ]
    subprocess.run(cmd, check=True, cwd=str(sbdd_bench_repo))  # noqa: S603

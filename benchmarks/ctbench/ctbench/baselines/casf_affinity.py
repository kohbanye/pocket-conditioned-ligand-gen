"""CASF affinity baselines (GenScore / Vina / Boltz-2): collect + rerun.

Collectors parse the sibling repos' existing per-complex outputs into the
canonical affinity schema (pdbid,logka,cluster,head). Rerun wrappers regenerate
them under the same protocol (GPU + the backend's env).
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path


def collect_genscore(baselines_repo: Path) -> pd.DataFrame:
    """GenScore scoring-power CSV (pdbid,logka,cluster,score) -> canonical schema."""
    df = pd.read_csv(baselines_repo / "casf_work" / "scoring_power_genscore.csv")
    return df.rename(columns={"score": "head"})[["pdbid", "logka", "cluster", "head"]]


def collect_vina(source_repo: Path) -> pd.DataFrame:
    """Vina scoring CSV (pdbid,logka,cluster,vina_score) -> canonical schema."""
    df = pd.read_csv(source_repo / "outputs" / "casf" / "vina_scoring.csv")
    return df.rename(columns={"vina_score": "head"})[
        ["pdbid", "logka", "cluster", "head"]
    ]


def rerun_genscore_scoring(
    baselines_repo: Path,
    extra_args: list[str] | None = None,
) -> None:
    """Re-run GenScore scoring power via ``run_casf_scoring.py`` (micromamba env)."""
    script = baselines_repo / "run_casf_scoring.py"
    env_dir = baselines_repo / "envs" / "genscore"
    cmd = [
        "micromamba",
        "run",
        "-p",
        str(env_dir),
        "python",
        str(script),
        "--backend",
        "genscore",
        *(extra_args or []),
    ]
    subprocess.run(cmd, check=True, cwd=str(baselines_repo))  # noqa: S603


def rerun_boltz_casf(source_repo: Path, extra_args: list[str] | None = None) -> None:
    """Re-run Boltz-2 affinity on CASF via the source repo's script (subprocess)."""
    script = source_repo / "scripts" / "eval_boltz_casf.py"
    cmd = ["python", str(script), *(extra_args or [])]
    subprocess.run(cmd, check=True, cwd=str(source_repo))  # noqa: S603

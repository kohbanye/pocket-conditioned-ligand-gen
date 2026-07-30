"""Standard per-sample dump schemas and validated loaders.

All inference (ours + baselines) writes per-sample dumps in these fixed schemas
so the metric/aggregation layer never needs to know which method produced them.
Column names match the source-repo dumps exactly so their existing CSVs can be
copied into ``results/`` and read unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

# CASF pose-rescoring per-pose dump (source: outputs/casf/pose_scores*.csv)
POSE_COLUMNS = ("pdbid", "pose", "rmsd", "head", "pll")
# CASF affinity per-complex dump (source: outputs/casf/affinity_*.csv)
AFFINITY_COLUMNS = ("pdbid", "logka", "cluster", "pll", "head")
# Generation per-molecule dump (subset of source tsweep_per_molecule.parquet)
GENERATION_MOLECULE_KEYS = ("model", "target_id")


def _require(df: pd.DataFrame, cols: tuple[str, ...], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        msg = f"{path}: missing required columns {missing}; have {list(df.columns)}"
        raise ValueError(msg)


def read_pose_scores(path: Path, method: str | None = None) -> pd.DataFrame:
    """Load a per-pose dump; optionally tag every row with a ``method`` column."""
    df = pd.read_csv(path)
    _require(df, ("pdbid", "pose", "rmsd"), path)
    if method is not None:
        df = df.assign(method=method)
    return df


def read_affinity(path: Path, method: str | None = None) -> pd.DataFrame:
    """Load a per-complex affinity dump; optionally tag with a ``method`` column."""
    df = pd.read_csv(path)
    _require(df, ("pdbid", "logka"), path)
    if method is not None:
        df = df.assign(method=method)
    return df


def read_generation_molecules(path: Path) -> pd.DataFrame:
    """Load a per-molecule generation dump (parquet or csv)."""
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    _require(df, ("model",), path)
    return df


def write_pose_scores(df: pd.DataFrame, path: Path) -> None:
    """Write a per-pose dump in the canonical column order."""
    _require(df, ("pdbid", "pose", "rmsd"), path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in POSE_COLUMNS if c in df.columns]
    df.to_csv(path, columns=cols, index=False)


def write_affinity(df: pd.DataFrame, path: Path) -> None:
    """Write a per-complex affinity dump in the canonical column order."""
    _require(df, ("pdbid", "logka"), path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in AFFINITY_COLUMNS if c in df.columns]
    df.to_csv(path, columns=cols, index=False)

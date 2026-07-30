"""Pose-rescoring metrics: CASF-2016 docking power and ranking (ported).

Mirrors ``notebooks/paper_pose_rescoring.py``. A "scored pose" frame has columns
``pdbid``, ``pose``, ``rmsd`` and a native-likeness column ``score`` where
*higher = more native-like*. Raw head/pll/Vina columns are first passed through
:func:`orient` so every method is on the same (higher-is-better) convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCORE = "score"
_MIN_POSES = 3


def orient(df: pd.DataFrame, raw_col: str = SCORE) -> pd.DataFrame:
    """Flip ``raw_col`` so higher = more native-like (anti-correlated with RMSD).

    Robust because every method here has a clear (|rho|>0.2) monotone signal
    against RMSD over the pooled pose set. Returns a copy with column ``score``.
    """
    out = df.rename(columns={raw_col: SCORE}).copy()
    raw = out[SCORE].to_numpy(dtype=float)
    rmsd = out["rmsd"].to_numpy(dtype=float)
    rho = spearmanr(raw, rmsd).correlation
    if rho is not None and rho > 0:  # higher raw <-> higher rmsd (worse) -> flip
        out[SCORE] = -raw
    return out


def docking_power(df: pd.DataFrame, cut: float) -> float:
    """Percent of targets whose top-scored pose is within ``cut`` Å of native."""
    ok = tot = 0
    for _, g in df.groupby("pdbid"):
        tot += 1
        if g.loc[g[SCORE].idxmax(), "rmsd"] < cut:
            ok += 1
    return 100.0 * ok / tot if tot else float("nan")


def target_success(df: pd.DataFrame, cut: float) -> pd.DataFrame:
    """Per-target 0/1 success (top-scored pose within ``cut`` Å), for paired tests."""
    rows = []
    for pdbid, g in df.groupby("pdbid"):
        hit = int(g.loc[g[SCORE].idxmax(), "rmsd"] < cut)
        rows.append({"pdbid": pdbid, "success": hit})
    return pd.DataFrame(rows)


def ranking_rho(df: pd.DataFrame) -> float:
    """Mean per-target Spearman rho, sign-flipped (high score should ↔ low RMSD)."""
    return float(np.mean(target_rho(df)["rho"])) if len(df) else float("nan")


def target_rho(df: pd.DataFrame) -> pd.DataFrame:
    """Per-target sign-flipped Spearman(score, rmsd) for clusters with >=3 poses."""
    rows = []
    for pdbid, g in df.groupby("pdbid"):
        if len(g) >= _MIN_POSES:
            r = spearmanr(g[SCORE], g["rmsd"]).correlation
            if r is not None and np.isfinite(r):
                rows.append({"pdbid": pdbid, "rho": -float(r)})
    return pd.DataFrame(rows)


def zsum(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Per-target z-score each frame's ``score`` then sum across frames (consensus).

    Frames are inner-joined on (pdbid, pose). Returns a scored frame with the
    fused ``score`` column, ready for :func:`docking_power` / :func:`ranking_rho`.
    """
    eps = 1e-9
    merged: pd.DataFrame | None = None
    for i, d in enumerate(frames):
        dd = d.copy()
        dd["z"] = dd.groupby("pdbid")[SCORE].transform(
            lambda s: (s - s.mean()) / (s.std() + eps),
        )
        dd = dd[["pdbid", "pose", "rmsd", "z"]].rename(columns={"z": f"z{i}"})
        merged = (
            dd
            if merged is None
            else merged.merge(
                dd.drop(columns="rmsd"),
                on=["pdbid", "pose"],
                how="inner",
            )
        )
    if merged is None:
        cols = ["pdbid", "pose", "rmsd", SCORE]
        return pd.DataFrame(columns=cols)  # ty: ignore[invalid-argument-type]
    zcols = [c for c in merged.columns if c.startswith("z")]
    merged[SCORE] = merged[zcols].sum(axis=1)
    return merged

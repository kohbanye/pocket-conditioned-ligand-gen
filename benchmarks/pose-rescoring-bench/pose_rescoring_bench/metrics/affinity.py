"""Affinity metrics: CASF-2016 scoring & ranking power (ported).

Mirrors ``notebooks/paper_affinity.py``. A prediction frame has columns
``logka`` (experimental pK), ``cluster`` (CASF target cluster id) and a
prediction column (default ``head``). Scoring power = Pearson R over all
complexes; ranking power = mean within-cluster Spearman rho over clusters with
>=3 members.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PRED = "head"
_MIN_CLUSTER = 3


def scoring_r(df: pd.DataFrame, pred: str = PRED) -> float:
    """Pearson R between experimental pK and prediction over all complexes."""
    d = df.dropna(subset=["logka", pred])
    if len(d) < 2:  # noqa: PLR2004
        return float("nan")
    return float(pearsonr(d["logka"], d[pred])[0])


def ranking_rho(df: pd.DataFrame, pred: str = PRED) -> float:
    """Mean within-cluster Spearman rho over clusters with >=3 members."""
    rs = cluster_rho(df, pred)["rho"].to_numpy()
    return float(np.mean(rs)) if rs.size else float("nan")


def cluster_rho(df: pd.DataFrame, pred: str = PRED) -> pd.DataFrame:
    """Per-cluster Spearman(logka, pred) for clusters with >=3 finite members."""
    d = df.dropna(subset=["logka", pred])
    rows = []
    for cid, g in d.groupby("cluster"):
        if len(g) >= _MIN_CLUSTER:
            r = spearmanr(g["logka"], g[pred]).correlation
            if r is not None and np.isfinite(r):
                rows.append({"cluster": cid, "rho": float(r)})
    return pd.DataFrame(rows)


def zsum_ensemble(frames: list[pd.DataFrame], pred: str = PRED) -> pd.DataFrame:
    """Fixed z-sum ensemble across heads, aligned on ``pdbid`` (no learned weights).

    Each frame is standardized on its prediction column and summed. Returns a
    frame with columns ``pdbid``, ``logka``, ``cluster`` and fused ``head``.
    """
    ens: pd.DataFrame | None = None
    for d in frames:
        dd = (
            d.dropna(subset=["logka", pred]).sort_values("pdbid").reset_index(drop=True)
        )
        if ens is None:
            ens = dd[["pdbid", "logka", "cluster"]].copy()
            ens[PRED] = 0.0
        h = dd[pred].to_numpy(dtype=float)
        ens[PRED] = ens[PRED].to_numpy() + (h - h.mean()) / h.std()
    if ens is None:
        cols = ["pdbid", "logka", "cluster", PRED]
        return pd.DataFrame(columns=cols)  # ty: ignore[invalid-argument-type]
    return ens

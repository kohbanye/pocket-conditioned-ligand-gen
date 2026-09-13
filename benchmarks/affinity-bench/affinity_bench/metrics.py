"""CASF-2016 scoring and ranking power.

A prediction frame has one row per core complex: ``pdbid``, ``logka`` (the
measured pK), ``cluster`` (the CASF target cluster the complex belongs to) and
a prediction column, ``head`` by default.

The two metrics answer different questions and a method can be good at one and
bad at the other. Scoring power is a single correlation over all 285 complexes,
so it is dominated by the range between weak and strong binders across
unrelated targets -- and a predictor that only knew ligand size would already do
well on it. Ranking power is the mean Spearman *inside* each five-ligand
cluster, where every ligand binds the same protein and the spread is often less
than one log unit; size carries no information there, so it is the metric that
asks whether the model resolves congeners.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PRED = "head"
#: CASF clusters hold five ligands; three is the floor at which a rank
#: correlation inside one is worth computing at all.
_MIN_CLUSTER = 3


def scoring_r(df: pd.DataFrame, pred: str = PRED) -> float:
    """Pearson R between the measured pK and the prediction, over all complexes."""
    d = df.dropna(subset=["logka", pred])
    if len(d) < 2:  # noqa: PLR2004
        return float("nan")
    return float(pearsonr(d["logka"], d[pred])[0])


def ranking_rho(df: pd.DataFrame, pred: str = PRED) -> float:
    """Mean within-cluster Spearman rho over clusters with enough members."""
    rs = cluster_rho(df, pred)["rho"].to_numpy()
    return float(np.mean(rs)) if rs.size else float("nan")


def cluster_rho(df: pd.DataFrame, pred: str = PRED) -> pd.DataFrame:
    """Per-cluster Spearman(logka, pred), for clusters with >=3 finite members.

    Returned per cluster rather than only averaged because the mean hides the
    shape of the failure: the published head scores 16 clusters perfectly and
    loses badly on nine, which is a different problem from being uniformly
    mediocre and wants a different fix.
    """
    d = df.dropna(subset=["logka", pred])
    rows = []
    for cid, g in d.groupby("cluster"):
        if len(g) >= _MIN_CLUSTER:
            r = spearmanr(g["logka"], g[pred]).correlation
            if r is not None and np.isfinite(r):
                rows.append({"cluster": cid, "rho": float(r)})
    # Columns are named even when no cluster survived, so a degenerate dump --
    # a head that collapsed to one value, say -- reads out as nan from
    # :func:`ranking_rho` instead of raising KeyError on a column-less frame.
    return pd.DataFrame(rows, columns=["cluster", "rho"])  # ty: ignore[invalid-argument-type]  # pandas stub: list[str] columns


def zsum_ensemble(frames: list[pd.DataFrame], pred: str = PRED) -> pd.DataFrame:
    """Fixed z-sum over several heads' dumps, aligned on ``pdbid``.

    Each frame is standardized on its prediction column and summed -- no learned
    weights and no member chosen by looking at the result, because a fusion
    fitted on the test set is not a measurement of the model. Used for ablation
    rows; the headline number is a single head.
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
        return pd.DataFrame(columns=["pdbid", "logka", "cluster", PRED])  # ty: ignore[invalid-argument-type]  # pandas stub: list[str] columns
    return ens

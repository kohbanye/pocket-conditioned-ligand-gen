"""Unit tests for affinity metrics (synthetic scoring/ranking sets)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ctbench.metrics import affinity as A


def _frame(logka: list[float], cluster: list[int], head: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pdbid": [f"p{i}" for i in range(len(logka))],
            "logka": logka,
            "cluster": cluster,
            "head": head,
        },
    )


def test_scoring_r_perfect_correlation() -> None:
    df = _frame([1.0, 2.0, 3.0, 4.0], [0, 0, 0, 0], [2.0, 4.0, 6.0, 8.0])
    assert abs(A.scoring_r(df) - 1.0) < 1e-9


def test_scoring_r_anticorrelation() -> None:
    df = _frame([1.0, 2.0, 3.0, 4.0], [0, 0, 0, 0], [4.0, 3.0, 2.0, 1.0])
    assert abs(A.scoring_r(df) + 1.0) < 1e-9


def test_ranking_rho_skips_small_clusters() -> None:
    # cluster 0 has 3 members (kept, perfect rank), cluster 1 has 2 (dropped)
    df = _frame([1.0, 2.0, 3.0, 5.0, 6.0], [0, 0, 0, 1, 1], [1.0, 2.0, 3.0, 9.0, 0.0])
    rho = A.ranking_rho(df)
    assert abs(rho - 1.0) < 1e-9  # only the well-ranked 3-member cluster counts


def test_cluster_rho_returns_one_row_per_kept_cluster() -> None:
    df = _frame(
        [1.0, 2.0, 3.0, 5.0, 6.0, 7.0],
        [0, 0, 0, 1, 1, 1],
        [1.0, 2.0, 3.0, 7.0, 6.0, 5.0],
    )
    cr = A.cluster_rho(df)
    assert set(cr["cluster"]) == {0, 1}
    assert abs(cr.set_index("cluster").loc[0, "rho"] - 1.0) < 1e-9
    assert abs(cr.set_index("cluster").loc[1, "rho"] + 1.0) < 1e-9


def test_zsum_ensemble_aligns_on_pdbid_and_standardizes() -> None:
    f1 = _frame([1.0, 2.0, 3.0], [0, 0, 0], [10.0, 20.0, 30.0])
    f2 = _frame([1.0, 2.0, 3.0], [0, 0, 0], [1.0, 2.0, 3.0])
    ens = A.zsum_ensemble([f1, f2])
    assert list(ens["pdbid"]) == ["p0", "p1", "p2"]
    assert abs(float(np.mean(ens["head"]))) < 1e-9  # z-sum is mean-centered
    assert abs(A.scoring_r(ens) - 1.0) < 1e-9  # both heads perfectly track logka

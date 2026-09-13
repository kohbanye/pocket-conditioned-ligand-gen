"""Unit tests for the two CASF metrics, on constructed frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from affinity_bench import metrics as A


def _frame(logka: list[float], pred: list[float], cluster: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pdbid": [f"c{i}" for i in range(len(logka))],
            "logka": logka,
            "cluster": cluster,
            "head": pred,
        },
    )


def test_scoring_r_is_pearson_over_all_complexes() -> None:
    df = _frame([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], ["a"] * 4)
    assert A.scoring_r(df) == pytest.approx(1.0)


def test_scoring_r_sees_sign() -> None:
    df = _frame([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], ["a"] * 4)
    assert A.scoring_r(df) == pytest.approx(-1.0)


def test_ranking_rho_averages_over_clusters_not_complexes() -> None:
    """A big cluster does not outvote a small one: the mean is over clusters."""
    df = _frame(
        # cluster a: ranked perfectly; cluster b: ranked exactly backwards
        [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1.0, 2.0, 3.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        ["a"] * 3 + ["b"] * 6,
    )
    assert A.ranking_rho(df) == pytest.approx(0.0)


def test_clusters_below_the_floor_are_dropped() -> None:
    df = _frame(
        [1.0, 2.0, 3.0, 1.0, 2.0],
        [1.0, 2.0, 3.0, 9.0, 0.0],
        ["a"] * 3 + ["b"] * 2,
    )
    assert list(A.cluster_rho(df)["cluster"]) == ["a"]
    assert A.ranking_rho(df) == pytest.approx(1.0)


def test_a_constant_prediction_leaves_no_cluster_to_score() -> None:
    """Spearman of a constant is nan; those clusters drop out rather than
    counting as zero, which would silently reward a model that predicts nothing."""
    df = _frame([1.0, 2.0, 3.0], [5.0, 5.0, 5.0], ["a"] * 3)
    assert A.cluster_rho(df).empty
    assert np.isnan(A.ranking_rho(df))


def test_scoring_r_needs_two_points() -> None:
    assert np.isnan(A.scoring_r(_frame([1.0], [1.0], ["a"])))


def test_zsum_ensemble_is_unweighted_and_order_free() -> None:
    a = _frame([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], ["a"] * 4)
    # Same ordering, a hundred times the scale: standardizing means it cannot
    # dominate the sum.
    b = _frame([1.0, 2.0, 3.0, 4.0], [100.0, 200.0, 300.0, 400.0], ["a"] * 4)
    ens = A.zsum_ensemble([a, b])
    assert A.scoring_r(ens) == pytest.approx(1.0)
    flipped = A.zsum_ensemble([b, a])
    assert ens["head"].to_numpy() == pytest.approx(flipped["head"].to_numpy())

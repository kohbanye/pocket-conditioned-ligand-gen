"""Unit tests for the statistics helpers (synthetic inputs, known answers)."""

from __future__ import annotations

import numpy as np
from prolit_bench import stats


def test_paired_ttest_detects_shift() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 50)
    res = stats.paired_ttest(base + 1.0, base)  # constant +1 shift
    assert res.n == 50
    assert res.pvalue < 1e-6
    assert "mean_diff=+1" in res.detail


def test_paired_ttest_no_difference_is_not_significant() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 40)
    res = stats.paired_ttest(x + rng.normal(0, 0.01, 40), x)
    assert res.pvalue > 0.05


def test_wilcoxon_paired_all_equal_returns_nan() -> None:
    x = np.array([0.1, 0.2, 0.3])
    res = stats.wilcoxon_paired(x, x.copy())
    assert np.isnan(res.pvalue)


def test_mcnemar_all_concordant_is_ns() -> None:
    j = np.array([1, 1, 0, 0])
    res = stats.mcnemar(j, j.copy())
    assert res.pvalue == 1.0


def test_mcnemar_one_sided_discordance() -> None:
    joint = np.array([1, 1, 1, 1, 1, 0])
    other = np.array([0, 0, 0, 0, 0, 0])  # joint wins 5, other wins 0
    res = stats.mcnemar(joint, other)
    assert res.pvalue < 0.1
    assert "joint_only=5" in res.detail


def test_compare_scoring_r_separates_good_from_bad() -> None:
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 200)
    good = y + rng.normal(0, 0.3, 200)  # strongly correlated
    bad = rng.normal(0, 1, 200)  # uncorrelated
    res = stats.compare_scoring_r(y, good, bad)
    assert res.pvalue < 0.01


def test_compare_scoring_r_ties_are_ns() -> None:
    rng = np.random.default_rng(3)
    y = rng.normal(0, 1, 200)
    a = y + rng.normal(0, 0.5, 200)
    b = y + rng.normal(0, 0.5, 200)
    res = stats.compare_scoring_r(y, a, b)
    assert res.pvalue > 0.05


def test_bootstrap_ci_brackets_mean() -> None:
    rng = np.random.default_rng(4)
    v = rng.normal(5.0, 1.0, 500)
    ci = stats.bootstrap_ci(v, seed=0)
    assert ci.low < ci.point < ci.high
    assert abs(ci.point - 5.0) < 0.2


def test_bootstrap_ci_paired_diff_excludes_zero_for_clear_shift() -> None:
    rng = np.random.default_rng(5)
    base = rng.normal(0, 1, 300)
    ci = stats.bootstrap_ci_paired_diff(base + 0.8, base, seed=0)
    assert ci.low > 0.0  # CI excludes 0 -> difference is real


def test_holm_correction_orders_and_bounds() -> None:
    raw = [0.01, 0.04, 0.03, 0.005]
    adj = stats.holm_correction(raw)
    assert all(a >= r for a, r in zip(adj, raw, strict=True))  # adjusted >= raw
    assert all(a <= 1.0 for a in adj)


def test_holm_correction_passes_nan_through() -> None:
    adj = stats.holm_correction([0.01, float("nan"), 0.02])
    assert np.isnan(adj[1])
    assert not np.isnan(adj[0])

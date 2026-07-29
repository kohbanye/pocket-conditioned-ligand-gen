"""Statistical significance helpers for benchmark comparisons.

The project's central question is not "which point estimate is higher" but
"is the difference significant" — the model neither clearly wins nor clearly
loses against existing methods, and the joint-vs-single tokenizer ablation must
be judged the same way. Every comparison therefore returns a p-value (and, where
meaningful, a confidence interval) rather than a bare delta.

Test choices mirror the source repo's paper notebooks:
- generation: paired t-test on the per-target metric (``scipy.stats.ttest_rel``);
- affinity scoring R: Williams/Steiger test for two dependent, overlapping
  correlations that share the experimental-pK variable;
- ranking rho / other per-unit paired metrics: Wilcoxon signed-rank;
- docking-power success (paired binary over targets): McNemar.
Multiple pairwise ablation comparisons are corrected with Holm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

_MIN_WILLIAMS_N = 4
_DEFAULT_BOOTSTRAP = 10_000


@dataclass(frozen=True)
class TestResult:
    """Outcome of a two-sided significance test."""

    statistic: float
    pvalue: float
    n: int
    detail: str = ""


@dataclass(frozen=True)
class CIResult:
    """A point estimate with a (bootstrap) confidence interval."""

    point: float
    low: float
    high: float
    level: float


def paired_ttest(a: np.ndarray, b: np.ndarray) -> TestResult:
    """Paired two-sided t-test of ``a`` vs ``b`` (e.g. per-target Vina score).

    ``a`` and ``b`` are aligned per unit (target/complex). Returns nan p-value
    when there are fewer than two pairs.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        msg = f"paired arrays must match: {a.shape} vs {b.shape}"
        raise ValueError(msg)
    n = int(a.size)
    if n < 2:  # noqa: PLR2004
        return TestResult(float("nan"), float("nan"), n, "n<2")
    res = sps.ttest_rel(a, b)
    mean_diff = float(np.mean(a - b))
    return TestResult(
        float(res.statistic),
        float(res.pvalue),
        n,
        f"mean_diff={mean_diff:+.4f}",
    )


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> TestResult:
    """Wilcoxon signed-rank test of paired ``a`` vs ``b`` (e.g. per-cluster rho)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    diff = diff[np.isfinite(diff)]
    n = int(diff.size)
    if n < 1 or np.allclose(diff, 0.0):
        return TestResult(float("nan"), float("nan"), n, "no non-zero differences")
    res = sps.wilcoxon(diff)
    return TestResult(
        float(res.statistic),
        float(res.pvalue),
        n,
        f"median_diff={float(np.median(diff)):+.4f}",
    )


def mcnemar(joint_success: np.ndarray, other_success: np.ndarray) -> TestResult:
    """Exact McNemar test for paired binary outcomes (docking-power hits).

    ``joint_success`` / ``other_success`` are 0/1 arrays aligned per target.
    Uses the exact binomial form on the discordant pairs.
    """
    j = np.asarray(joint_success, dtype=int)
    o = np.asarray(other_success, dtype=int)
    if j.shape != o.shape:
        msg = f"paired arrays must match: {j.shape} vs {o.shape}"
        raise ValueError(msg)
    b = int(np.sum((j == 1) & (o == 0)))  # joint wins
    c = int(np.sum((j == 0) & (o == 1)))  # other wins
    n_disc = b + c
    if n_disc == 0:
        return TestResult(0.0, 1.0, int(j.size), "no discordant pairs")
    pval = float(sps.binomtest(min(b, c), n_disc, 0.5).pvalue)
    return TestResult(
        float(b - c),
        pval,
        int(j.size),
        f"joint_only={b}, other_only={c}",
    )


def williams_test(r_xy: float, r_xz: float, r_yz: float, n: int) -> TestResult:
    """Williams/Steiger test that dependent correlations ``r_xy`` and ``r_xz`` differ.

    ``x`` is the shared variable (experimental pK); ``y``/``z`` are the two
    predictions. ``r_yz`` is the correlation between the two predictions. This is
    the test used for CASF scoring-power head-to-head comparison.
    """
    if n < _MIN_WILLIAMS_N:
        return TestResult(float("nan"), float("nan"), n, "n too small")
    num = (r_xy - r_xz) * np.sqrt((n - 1) * (1 + r_yz))
    det = 1.0 - r_xy**2 - r_xz**2 - r_yz**2 + 2.0 * r_xy * r_xz * r_yz
    den = np.sqrt(
        2.0 * ((n - 1) / (n - 3)) * det + ((r_xy + r_xz) ** 2 / 4.0) * (1 - r_yz) ** 3,
    )
    if den == 0:
        return TestResult(float("nan"), float("nan"), n, "degenerate denominator")
    t = float(num / den)
    pval = float(2.0 * sps.t.sf(abs(t), df=n - 3))
    return TestResult(t, pval, n, f"r_xy={r_xy:.4f}, r_xz={r_xz:.4f}")


def compare_scoring_r(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> TestResult:
    """Williams test comparing Pearson R(y, pred_a) vs R(y, pred_b) on shared ``y``."""
    y = np.asarray(y, dtype=float)
    a = np.asarray(pred_a, dtype=float)
    b = np.asarray(pred_b, dtype=float)
    mask = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    y, a, b = y[mask], a[mask], b[mask]
    n = int(y.size)
    if n < _MIN_WILLIAMS_N:
        return TestResult(float("nan"), float("nan"), n, "n too small")
    r_xy = float(np.corrcoef(y, a)[0, 1])
    r_xz = float(np.corrcoef(y, b)[0, 1])
    r_yz = float(np.corrcoef(a, b)[0, 1])
    return williams_test(r_xy, r_xz, r_yz, n)


def bootstrap_ci(
    values: np.ndarray,
    *,
    level: float = 0.95,
    n_resamples: int = _DEFAULT_BOOTSTRAP,
    seed: int = 0,
) -> CIResult:
    """Percentile bootstrap CI for the mean of ``values`` (e.g. per-molecule Vina)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return CIResult(float("nan"), float("nan"), float("nan"), level)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_resamples, v.size))
    means = v[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return CIResult(float(v.mean()), float(low), float(high), level)


def bootstrap_ci_paired_diff(
    a: np.ndarray,
    b: np.ndarray,
    *,
    level: float = 0.95,
    n_resamples: int = _DEFAULT_BOOTSTRAP,
    seed: int = 0,
) -> CIResult:
    """Bootstrap CI for the mean paired difference ``a - b`` (resamples units)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return CIResult(float("nan"), float("nan"), float("nan"), level)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_resamples, diff.size))
    means = diff[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return CIResult(float(diff.mean()), float(low), float(high), level)


def holm_correction(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, order-preserving.

    NaN inputs pass through as NaN and are excluded from the correction count.
    """
    arr = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(arr)
    out = np.full(arr.shape, np.nan)
    idx = np.where(finite)[0]
    if idx.size == 0:
        return out.tolist()
    order = idx[np.argsort(arr[idx])]
    m = idx.size
    running = 0.0
    for rank, i in enumerate(order):
        adj = (m - rank) * arr[i]
        running = max(running, adj)
        out[i] = min(running, 1.0)
    return out.tolist()

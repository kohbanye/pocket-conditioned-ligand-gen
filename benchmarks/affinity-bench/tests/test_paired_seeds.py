"""Arms are compared seed by seed, because a single run is a draw, not a value.

Retraining this head with nothing changed but the seed moves scoring R by about
0.05 and ranking rho by about 0.04 -- larger than any architectural difference
measured so far. Comparing one run of an arm against one run of another is
therefore a coin flip dressed as a result, and it produced a conclusion that had
to be retracted. Pairing on the seed cancels the shared term.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from affinity_bench import aggregate


def _frame(pred: list[float], logka: list[float], cluster: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pdbid": [f"c{i}" for i in range(len(pred))],
            "logka": logka,
            "cluster": cluster,
            "head": pred,
        },
    )


def _arm(offsets: dict[int, float]) -> dict[int, pd.DataFrame]:
    """One arm across seeds: a seed shifts the prediction of one complex."""
    logka = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    cluster = ["a"] * 3 + ["b"] * 3
    return {
        seed: _frame([1.0, 2.0, 3.0 + off, 4.0, 5.0, 6.0], logka, cluster)
        for seed, off in offsets.items()
    }


def test_seed_spread_reports_sd_not_a_single_number() -> None:
    spread = aggregate.seed_spread(_arm({7: 0.0, 11: 2.0, 12: -2.0}))
    assert spread["n_seeds"] == 3
    assert spread["scoring_R_sd"] > 0, "a spread of runs must not read as exact"


def test_a_single_seed_reports_nan_sd_rather_than_zero() -> None:
    """One run has no spread to report; zero would claim a precision it lacks."""
    spread = aggregate.seed_spread(_arm({7: 0.0}))
    assert spread["n_seeds"] == 1
    assert np.isnan(spread["scoring_R_sd"])


def test_pairing_cancels_the_seed_and_finds_a_shared_effect() -> None:
    """An arm that is better on every seed reads as better, despite the noise.

    The seed is modelled as a displacement of one complex's prediction, and it
    is large and signed: unpaired, the arms' individual spreads overlap
    completely. The better arm halves that displacement on every seed, so the
    paired difference is positive three times out of three even though neither
    arm's mean is separable from the other's.
    """
    noise = {7: 1.0, 11: 3.0, 12: -3.0}
    ref = _arm(noise)
    better = _arm({s: o / 2 for s, o in noise.items()})

    out = aggregate.paired_by_seed({"ref": ref, "better": better}, reference="ref")
    assert out.loc["better", "n_seeds"] == 3
    assert out.loc["better", "scoring_n_better"] == 3, "all three seeds must agree"
    assert out.loc["better", "d_scoring_R"] > 0
    assert out.loc["better", "d_ranking_rho"] > 0


def test_only_shared_seeds_are_compared() -> None:
    ref = _arm({7: 0.0, 11: 1.0})
    other = _arm({11: 1.0, 12: 5.0})
    out = aggregate.paired_by_seed({"ref": ref, "other": other}, reference="ref")
    assert out.loc["other", "n_seeds"] == 1, "only seed 11 is in both"


def test_an_arm_with_no_shared_seed_is_dropped_not_compared_unpaired() -> None:
    ref = _arm({7: 0.0})
    other = _arm({99: 0.0})
    out = aggregate.paired_by_seed({"ref": ref, "other": other}, reference="ref")
    assert out.empty


def test_n_better_splits_are_visible_when_the_effect_is_not_real() -> None:
    """2-of-3 is the signature of noise, and must be readable as such."""
    ref = _arm({7: 0.0, 11: 0.0, 12: 0.0})
    # Two seeds move one way, the third the other: exactly what a real effect
    # does not look like.
    mixed = _arm({7: -0.1, 11: -0.1, 12: 0.9})
    out = aggregate.paired_by_seed({"ref": ref, "mixed": mixed}, reference="ref")
    assert out.loc["mixed", "scoring_n_better"] < 3

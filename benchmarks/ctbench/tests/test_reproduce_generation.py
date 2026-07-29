"""Reproduce the source repo's generation comparison + paired significance.

Targets = ``docs/comparison_tables.md`` (Vina score/min per method) and
``notebooks/paper_generation.py`` (paired t-test vs DiffGui over 3 targets).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ctbench.aggregate import generation_pairwise, generation_table

_RESULTS = Path(__file__).resolve().parent.parent / "results" / "generation"

pytestmark = pytest.mark.skipif(
    not (_RESULTS / "joint").exists(),
    reason="generation dumps not seeded",
)


def _per_model() -> dict[str, pd.Series]:
    ours = pd.read_csv(_RESULTS / "joint" / "per_model.csv").set_index("model")
    base = pd.read_csv(_RESULTS / "baselines" / "per_model.csv").set_index("model")
    return {
        "diffgui": base.loc["diffgui"],
        "targetdiff": base.loc["targetdiff"],
        "diffsbdd": base.loc["diffsbdd"],
        "OURS": ours.loc["own_t085_on"],
    }


def test_generation_vina_table() -> None:
    table = generation_table(_per_model())
    assert table.loc["diffgui", "vina_score_mean"] == pytest.approx(-6.54, abs=0.02)
    assert table.loc["targetdiff", "vina_score_mean"] == pytest.approx(-4.76, abs=0.02)
    assert table.loc["diffsbdd", "vina_score_mean"] == pytest.approx(-4.40, abs=0.02)
    assert table.loc["OURS", "vina_score_mean"] == pytest.approx(-5.33, abs=0.02)
    assert table.loc["OURS", "vina_min_mean"] == pytest.approx(-6.92, abs=0.02)


def test_generation_paired_ttest_vs_diffgui_not_significant() -> None:
    ours = pd.read_csv(_RESULTS / "joint" / "per_target.csv")
    ours = ours[ours.model == "own_t085_on"]
    base = pd.read_csv(_RESULTS / "baselines" / "per_target.csv")
    per_target = {"OURS": ours, "diffgui": base[base.model == "diffgui"]}
    sig = generation_pairwise(per_target, reference="diffgui", metric="vina_score_mean")
    assert sig.loc["OURS", "ttest_p"] > 0.05  # n=3 -> underpowered, not significant

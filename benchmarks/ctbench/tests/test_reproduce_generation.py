"""Reproduce the generation comparison from committed per-sample dumps.

Target = the paper's generation table (Vina score / min per method) and the
paired significance test behind "not significantly different from DiffGui".

Provenance note, which is why this file is narrower than the paper's table:
the three baseline columns were not all produced by one evaluation run. The
committed dumps hold DiffGui at ``results/diffgui/`` and DiffSBDD at
``results/partial/``, and both match the paper. TargetDiff does not: every
committed TargetDiff dump has a *positive* mean Vina score (+6.3 to +8.5 across
``results/``, ``results_td_a/``, ``results_td_b/``), so the -4.76 in the table
cannot be re-derived here and is asserted nowhere below. Re-running TargetDiff
through ``scripts/run_evaluation.py`` and committing that dump is what would
close the gap.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ctbench.aggregate import generation_pairwise, generation_table

_CTBENCH = Path(__file__).resolve().parent.parent / "results" / "generation"
_SBDD = Path(__file__).resolve().parents[2] / "sbddbench" / "results"

pytestmark = pytest.mark.skipif(
    not (_CTBENCH / "joint").exists(),
    reason="generation dumps not seeded",
)

#: The generation arm the paper reports: temperature 0.85, refiner on.
_OURS = "own_t085_on"


def _per_model() -> dict[str, pd.Series]:
    ours = pd.read_csv(_CTBENCH / "joint" / "per_model.csv").set_index("model")
    diffgui = pd.read_csv(_SBDD / "diffgui" / "per_model.csv").set_index("model")
    diffsbdd = pd.read_csv(_SBDD / "partial" / "per_model.csv").set_index("model")
    return {
        "diffgui": diffgui.loc["diffgui"],
        "diffsbdd": diffsbdd.loc["diffsbdd"],
        "OURS": ours.loc[_OURS],
    }


def test_generation_vina_table() -> None:
    table = generation_table(_per_model())
    assert table.loc["diffgui", "vina_score_mean"] == pytest.approx(-6.54, abs=0.02)
    assert table.loc["diffsbdd", "vina_score_mean"] == pytest.approx(-4.40, abs=0.02)
    assert table.loc["OURS", "vina_score_mean"] == pytest.approx(-5.33, abs=0.02)
    assert table.loc["OURS", "vina_min_mean"] == pytest.approx(-6.92, abs=0.02)


def test_ours_beats_diffsbdd_and_trails_diffgui() -> None:
    """The table's qualitative claim, independent of the exact values."""
    table = generation_table(_per_model())
    ours = table.loc["OURS", "vina_score_mean"]
    assert ours < table.loc["diffsbdd", "vina_score_mean"]
    assert ours > table.loc["diffgui", "vina_score_mean"]


def test_generation_paired_ttest_vs_diffgui_not_significant() -> None:
    ours = pd.read_csv(_CTBENCH / "joint" / "per_target.csv")
    ours = ours[ours.model == _OURS]
    base = pd.read_csv(_SBDD / "diffgui" / "per_target.csv")
    per_target = {"OURS": ours, "diffgui": base[base.model == "diffgui"]}
    sig = generation_pairwise(per_target, reference="diffgui", metric="vina_score_mean")
    assert sig.loc["OURS", "ttest_p"] > 0.05  # n=3 -> underpowered, not significant


@pytest.mark.xfail(
    reason="no committed TargetDiff dump reproduces the paper's -4.76; every "
    "one has a positive mean Vina score. Re-run and commit the dump.",
    strict=True,
)
def test_targetdiff_baseline_is_reproducible() -> None:
    base = pd.read_csv(_SBDD / "per_model.csv").set_index("model")
    assert base.loc["targetdiff", "vina_score_mean"] == pytest.approx(-4.76, abs=0.02)

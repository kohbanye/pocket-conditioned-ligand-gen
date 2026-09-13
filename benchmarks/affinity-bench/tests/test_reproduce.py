"""Reproduce the recorded CASF numbers from the local dumps.

These pin the analysis layer against real data: if a metric or a loader drifts,
a published number moves and this says so. Dumps are not tracked in git (the
repo carries code), so the module skips when they are not on this machine.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from affinity_bench import metrics as A
from affinity_bench import report
from affinity_bench.aggregate import affinity_metrics, affinity_pairwise

_RESULTS = Path(__file__).resolve().parent.parent / "results"
_OURS = _RESULTS / "v1_kdki_mean" / "kdki-mean.csv"

pytestmark = pytest.mark.skipif(
    not _OURS.exists(),
    reason="affinity dumps are not present locally (results/ is git-ignored)",
)


def _ours() -> pd.DataFrame:
    return pd.read_csv(_OURS)


def _genscore() -> pd.DataFrame:
    return pd.read_csv(_RESULTS / "genscore" / "scoring.csv").rename(
        columns={"score": "head"},
    )


def test_published_single_head_numbers() -> None:
    """The arm the paper would report today: one mean-pooled Kd/Ki head."""
    ours = _ours()
    assert A.scoring_r(ours) == pytest.approx(0.7711, abs=1e-3)
    assert A.ranking_rho(ours) == pytest.approx(0.6474, abs=1e-3)


def test_genscore_reference_numbers() -> None:
    gen = _genscore()
    assert A.scoring_r(gen) == pytest.approx(0.8159, abs=1e-3)
    assert A.ranking_rho(gen) == pytest.approx(0.7351, abs=1e-3)


def test_vina_scoring_power() -> None:
    vina = pd.read_csv(_RESULTS / "vina" / "scoring.csv")
    col = "head" if "head" in vina.columns else "vina_score"
    assert abs(A.scoring_r(vina, col)) == pytest.approx(0.6076, abs=2e-3)


def test_boltz2_reference_numbers() -> None:
    """Collected from the run tree, not from a CSV, so this pins the parse too."""
    path = _RESULTS / "boltz2" / "scoring.csv"
    if not path.exists():
        pytest.skip("Boltz-2 predictions are not on this machine")
    boltz = pd.read_csv(path)
    assert A.scoring_r(boltz) == pytest.approx(0.7532, abs=1e-3)
    assert A.ranking_rho(boltz) == pytest.approx(0.7158, abs=1e-3)


def test_we_do_not_yet_beat_genscore() -> None:
    """Recorded as the current fact, and as the thing the loop has to change.

    When this fails because both deltas turned positive, that IS the result --
    update the numbers above and say so in ``docs/results/``.
    """
    preds = {"OURS": _ours(), "GenScore": _genscore()}
    sig = affinity_pairwise(preds, reference="GenScore")
    assert sig.loc["OURS", "d_scoring_R"] < 0
    assert sig.loc["OURS", "d_ranking_rho"] < 0


def test_metrics_table_assembles_from_the_dump_tree() -> None:
    metrics, sig = report.comparison(_RESULTS)
    assert "GenScore" in metrics.index
    assert metrics.loc["GenScore", "scoring_R"] == pytest.approx(0.8159, abs=1e-3)
    assert not sig.empty


def test_scoreboard_states_the_goal() -> None:
    board = report.scoreboard(_RESULTS)
    assert not board.empty
    assert not board["beats_both"].any()


def test_affinity_metrics_counts_the_complexes_it_used() -> None:
    table = affinity_metrics({"OURS": _ours()})
    assert table.loc["OURS", "n"] == len(_ours().dropna(subset=["logka", "head"]))

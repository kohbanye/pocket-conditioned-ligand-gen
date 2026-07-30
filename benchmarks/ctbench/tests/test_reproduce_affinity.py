"""Reproduce the source repo's known CASF affinity numbers from copied dumps.

Guards the metric implementations against the real data: the LF5 z-sum ensemble
must reproduce OUR paper number and GenScore/Vina must reproduce theirs, matching
``outputs/casf/method_comparison.csv`` (copied to ``results/affinity/``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ctbench.aggregate import affinity_metrics
from ctbench.metrics import affinity as A

_RESULTS = Path(__file__).resolve().parent.parent / "results" / "affinity"
_JOINT = _RESULTS / "joint"
_LF5 = (
    "mean_ic50.csv",
    "attn_ic50.csv",
    "mean_kdki.csv",
    "attn_kdki.csv",
    "meanmax_kdki.csv",
)

# Result dumps are not tracked in git -- only code is.
pytestmark = pytest.mark.skipif(
    not _JOINT.exists(),
    reason="affinity dumps not present locally (results/ is git-ignored)",
)


def _lf5_ensemble() -> pd.DataFrame:
    frames = [pd.read_csv(_JOINT / f) for f in _LF5]
    return A.zsum_ensemble(frames)


def test_our_lf5_ensemble_matches_paper() -> None:
    ens = _lf5_ensemble()
    assert A.scoring_r(ens) == pytest.approx(0.7899, abs=1e-3)
    assert A.ranking_rho(ens) == pytest.approx(0.6737, abs=1e-3)


def test_genscore_matches_method_comparison() -> None:
    gen = pd.read_csv(_RESULTS / "genscore" / "scoring.csv")
    assert A.scoring_r(gen, "score") == pytest.approx(0.8159, abs=1e-3)
    assert A.ranking_rho(gen, "score") == pytest.approx(0.7351, abs=1e-3)


def test_vina_scoring_power_matches_reference() -> None:
    vina = pd.read_csv(_RESULTS / "vina" / "scoring.csv")
    # method_comparison reports Vina scoring_R = 0.6076 (magnitude; sign aside)
    assert abs(A.scoring_r(vina, "vina_score")) == pytest.approx(0.6076, abs=2e-3)


def test_affinity_metrics_table_assembles() -> None:
    preds = {
        "OURS (LF5)": _lf5_ensemble(),
        "GenScore": pd.read_csv(_RESULTS / "genscore" / "scoring.csv").rename(
            columns={"score": "head"},
        ),
    }
    table = affinity_metrics(preds)
    assert table.loc["GenScore", "scoring_R"] == pytest.approx(0.8159, abs=1e-3)
    assert table.loc["OURS (LF5)", "scoring_R"] == pytest.approx(0.7899, abs=1e-3)

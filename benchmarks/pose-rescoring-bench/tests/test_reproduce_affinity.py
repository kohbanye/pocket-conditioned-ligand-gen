"""Reproduce the CASF affinity numbers from the local dumps.

Guards the metric implementations against real data. The arm is a SINGLE
mean-pooled head: this file used to pin a five-member z-sum ensemble over
``{mean,attn,meanmax} x {ic50,kdki}`` heads, but the attn and meanmax readouts
were removed after each was measured and neither beat the plain ligand-mean, so
those dumps moved to ``docs/results/stale_prefix_2026-08-25/``.

Regenerated with the fixed V2000 counts-line parser, which changes the five
CASF ligands that carry 100+ atoms or bonds (3ag9, 3bv9, 3prs, 3pww, 3uri).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pose_rescoring_bench.aggregate import affinity_metrics
from pose_rescoring_bench.metrics import affinity as A

_RESULTS = Path(__file__).resolve().parent.parent / "results" / "affinity"
_JOINT = _RESULTS / "joint" / "kdki-mean.csv"

# Result dumps are not tracked in git -- only code is.
pytestmark = pytest.mark.skipif(
    not _JOINT.exists(),
    reason="affinity dumps not present locally (results/ is git-ignored)",
)


def _ours() -> pd.DataFrame:
    return pd.read_csv(_JOINT)


def test_our_head_matches_recorded_numbers() -> None:
    ours = _ours()
    assert A.scoring_r(ours) == pytest.approx(0.7711, abs=1e-3)
    assert A.ranking_rho(ours) == pytest.approx(0.6474, abs=1e-3)


def test_affinity_still_trails_genscore() -> None:
    """Recorded as a fact, not a target: the pose head leads, this one does not."""
    gen = pd.read_csv(_RESULTS / "genscore" / "scoring.csv")
    assert A.scoring_r(_ours()) < A.scoring_r(gen, "score")


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
        "OURS (joint)": _ours(),
        "GenScore": pd.read_csv(_RESULTS / "genscore" / "scoring.csv").rename(
            columns={"score": "head"},
        ),
    }
    table = affinity_metrics(preds)
    assert table.loc["GenScore", "scoring_R"] == pytest.approx(0.8159, abs=1e-3)
    assert table.loc["OURS (joint)", "scoring_R"] == pytest.approx(
        0.7711, abs=1e-3
    )

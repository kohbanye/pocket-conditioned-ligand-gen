"""Reproduce the CASF pose docking-power numbers from the local dumps.

Every arm here is a SINGLE mean-pooled head. The earlier version of this file
pinned a three-head z-sum ensemble (``v2_mean`` + ``v6_meanmax`` + ``v7_attn``)
and a Vina-fused variant; those readouts were removed once each was measured
and none beat the plain ligand-mean, so their dumps moved to
``docs/results/stale_prefix_2026-08-25/``.

All five rows below were produced with the fixed V2000 counts-line parser (see
``prolit.tokenizers.ligand._counts_line``). Five CASF ligands -- 3ag9, 3bv9,
3prs, 3pww, 3uri -- have 100+ atoms or bonds and were parsed with three times
too many atoms before that fix, which blew the extracted pocket past the
training context and cost every one of them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pose_rescoring_bench.metrics import rescoring as R

_RESULTS = Path(__file__).resolve().parent.parent / "results" / "rescoring"

# Result dumps are not tracked in git -- only code is.
pytestmark = pytest.mark.skipif(
    not (_RESULTS / "e250_div" / "div_f16.csv").exists(),
    reason="pose dumps not present locally (results/ is git-ignored)",
)

_DP = 0.3  # docking-power tolerance (pt); table rounded to 0.1
_RHO = 0.02  # ranking-rho tolerance


def _our(variant: str, name: str) -> pd.DataFrame:
    return R.orient(pd.read_csv(_RESULTS / variant / name), raw_col="head")


def _baseline(backend: str, rmsd_key: pd.DataFrame) -> pd.DataFrame:
    d = pd.read_csv(_RESULTS / backend / "pose_scores.csv").merge(
        rmsd_key,
        on=["pdbid", "pose"],
        how="inner",
    )
    return R.orient(d, raw_col="native_score")


def test_pose_docking_power_table() -> None:
    div = _our("e250_div", "div_f16.csv")
    joint = _our("joint", "v2_f16.csv")
    vina = R.orient(
        pd.read_csv(_RESULTS / "vina" / "pose_scores.csv"), raw_col="head"
    )
    rmsd_key = div[["pdbid", "pose", "rmsd"]]
    rtm, gen = _baseline("rtmscore", rmsd_key), _baseline("genscore", rmsd_key)

    expect = {
        "RTMScore": (rtm, 94.4, 75.7, 0.86),
        "GenScore": (gen, 91.2, 73.6, 0.85),
        "Vina": (vina, 84.6, 71.6, 0.31),
        "joint": (joint, 89.5, 73.7, 0.85),
        "e250_div": (div, 95.4, 82.1, 0.89),
    }
    for label, (df, dp2, dp1, rho) in expect.items():
        assert R.docking_power(df, 2.0) == pytest.approx(dp2, abs=_DP), f"{label} DP@2"
        assert R.docking_power(df, 1.0) == pytest.approx(dp1, abs=_DP), f"{label} DP@1"
        assert R.ranking_rho(df) == pytest.approx(rho, abs=_RHO), f"{label} rho"


def test_e250_div_leads_every_metric() -> None:
    """The arm the paper reports beats RTMScore on all three, not just on one."""
    div = _our("e250_div", "div_f16.csv")
    rtm = _baseline("rtmscore", div[["pdbid", "pose", "rmsd"]])
    assert R.docking_power(div, 2.0) > R.docking_power(rtm, 2.0)
    assert R.docking_power(div, 1.0) > R.docking_power(rtm, 1.0)
    assert R.ranking_rho(div) > R.ranking_rho(rtm)

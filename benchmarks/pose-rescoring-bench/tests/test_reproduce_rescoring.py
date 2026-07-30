"""Reproduce the source repo's CASF pose docking-power table from copied dumps.

Targets = the published docking-power table (decoys-only, shared 284-target set).
Confirms orient/docking_power/ranking_rho/zsum reproduce every published row.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pose_rescoring_bench.metrics import rescoring as R

_RESULTS = Path(__file__).resolve().parent.parent / "results" / "rescoring"

# Result dumps are not tracked in git -- only code is.
pytestmark = pytest.mark.skipif(
    not (_RESULTS / "joint").exists(),
    reason="pose dumps not present locally (results/ is git-ignored)",
)

_DP = 0.3  # docking-power tolerance (pt); table rounded to 0.1
_RHO = 0.02  # ranking-rho tolerance


def _our(name: str) -> pd.DataFrame:
    return R.orient(pd.read_csv(_RESULTS / "joint" / name), raw_col="head")


def _vina() -> pd.DataFrame:
    return R.orient(pd.read_csv(_RESULTS / "vina" / "pose_scores.csv"), raw_col="head")


def _baseline(backend: str, rmsd_key: pd.DataFrame) -> pd.DataFrame:
    d = pd.read_csv(_RESULTS / backend / "pose_scores.csv").merge(
        rmsd_key,
        on=["pdbid", "pose"],
        how="inner",
    )
    return R.orient(d, raw_col="native_score")


def test_pose_docking_power_table() -> None:
    v2, v6, v7, vina = (
        _our("v2_mean.csv"),
        _our("v6_meanmax.csv"),
        _our("v7_attn.csv"),
        _vina(),
    )
    rmsd_key = v2[["pdbid", "pose", "rmsd"]]
    rtm, gen = _baseline("rtmscore", rmsd_key), _baseline("genscore", rmsd_key)
    ens3 = R.zsum([v2, v6, v7])
    ens_v2_vina = R.zsum([v2, vina])
    ens3_vina = R.zsum([v2, v6, v7, vina])

    expect = {
        "RTMScore": (rtm, 94.0, 75.4, 0.86),
        "GenScore": (gen, 90.8, 73.2, 0.85),
        "Vina": (vina, 84.6, 71.6, 0.31),
        "v2": (v2, 88.8, 66.7, 0.83),
        "3-head": (ens3, 89.5, 70.9, 0.83),
        "v2+Vina": (ens_v2_vina, 89.8, 77.9, 0.68),
        "3-head+Vina": (ens3_vina, 90.5, 75.8, 0.80),
    }
    for label, (df, dp2, dp1, rho) in expect.items():
        assert R.docking_power(df, 2.0) == pytest.approx(dp2, abs=_DP), f"{label} DP@2"
        assert R.docking_power(df, 1.0) == pytest.approx(dp1, abs=_DP), f"{label} DP@1"
        assert R.ranking_rho(df) == pytest.approx(rho, abs=_RHO), f"{label} rho"

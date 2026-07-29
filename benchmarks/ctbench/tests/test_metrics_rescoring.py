"""Unit tests for pose-rescoring metrics (synthetic pose sets)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ctbench.metrics import rescoring as R


def _target(pdbid: str, rmsds: list[float], scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pdbid": pdbid,
            "pose": [f"{pdbid}_{i}" for i in range(len(rmsds))],
            "rmsd": rmsds,
            R.SCORE: scores,
        },
    )


def test_docking_power_counts_top_score_within_cutoff() -> None:
    # A: top score at rmsd 0.5 -> hit; B: top score at rmsd 3 -> miss
    a = _target("a", [0.5, 2.0, 4.0], [5.0, 3.0, 1.0])
    b = _target("b", [3.0, 1.5, 0.8], [9.0, 2.0, 1.0])
    df = pd.concat([a, b], ignore_index=True)
    assert R.docking_power(df, 2.0) == 50.0
    assert R.docking_power(df, 1.0) == 50.0  # A still hit (0.5<1), B still miss


def test_target_success_is_per_target_binary() -> None:
    a = _target("a", [0.5, 2.0], [5.0, 1.0])
    b = _target("b", [3.0, 1.0], [9.0, 1.0])
    succ = R.target_success(pd.concat([a, b], ignore_index=True), 2.0).set_index(
        "pdbid",
    )["success"]
    assert succ["a"] == 1
    assert succ["b"] == 0


def test_ranking_rho_positive_when_score_tracks_native() -> None:
    # score decreases as rmsd increases -> strong native-likeness signal -> rho ~ +1
    df = _target("a", [0.5, 1.0, 2.0, 4.0], [4.0, 3.0, 2.0, 1.0])
    assert R.ranking_rho(df) > 0.9


def test_orient_flips_when_raw_is_positively_correlated_with_rmsd() -> None:
    # raw grows with rmsd (worse) -> orient() must flip so higher = better
    df = pd.DataFrame(
        {
            "pdbid": "a",
            "pose": ["a_0", "a_1", "a_2"],
            "rmsd": [0.5, 2.0, 4.0],
            "raw": [1.0, 2.0, 3.0],
        },
    )
    out = R.orient(df, raw_col="raw")
    # after orientation the top-scored pose is the low-rmsd one
    assert out.loc[out[R.SCORE].idxmax(), "rmsd"] == 0.5


def test_zsum_fuses_two_frames() -> None:
    a = _target("a", [0.5, 2.0, 4.0], [4.0, 3.0, 1.0])
    b = a.copy()
    fused = R.zsum([a, b])
    assert set(fused["pdbid"]) == {"a"}
    # fused top pose still the near-native one
    assert fused.loc[fused[R.SCORE].idxmax(), "rmsd"] == 0.5
    assert np.isfinite(fused[R.SCORE]).all()

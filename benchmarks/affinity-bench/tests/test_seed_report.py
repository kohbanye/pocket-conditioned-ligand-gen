"""The reported table is per family across seeds, not per run.

A family is a set of arms that are the same recipe at a different TRAINING seed.
Retraining moves scoring R by ~0.05, so a single run is a draw from a
distribution, not a value, and the table has to show the spread.

These tests inject their own roster rather than reading the live one. The live
roster changes for real reasons -- a contaminated family gets pulled, a new one
gets added -- and a test coupled to it fails for those changes instead of for
regressions. It did exactly that once. Whether the live roster is internally
consistent is a separate question, covered by ``test_registry_contract.py``.

``report`` imports ``SEED_FAMILIES`` and ``SEED_REFERENCE`` by name, so the
patch has to land in ``report``'s namespace; patching ``variants`` would rebind
nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from affinity_bench import report

if TYPE_CHECKING:
    from pathlib import Path

#: A two-family roster standing in for the real one.
ROSTER = {
    "ctrl": {7: "ctrl", 11: "ctrl_s11", 12: "ctrl_s12"},
    "cand": {7: "cand", 11: "cand_s11", 12: "cand_s12"},
}
REFERENCE = "ctrl"


@pytest.fixture(autouse=True)
def _roster(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report, "SEED_FAMILIES", ROSTER)
    monkeypatch.setattr(report, "SEED_REFERENCE", REFERENCE)


def _dump(path: Path, preds: list[float]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "pdbid": [f"c{i}" for i in range(len(preds))],
            "logka": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "cluster": ["a"] * 3 + ["b"] * 3,
            "head": preds,
        },
    ).to_csv(path / "kdki-mean.csv", index=False)


@pytest.fixture
def results(tmp_path: Path) -> Path:
    # The seed displaces one complex's prediction; cand halves that displacement.
    for arm, off in (
        ("ctrl_f16", 1.0), ("ctrl_s11_f16", 3.0), ("ctrl_s12_f16", -3.0),
        ("cand_f16", 0.5), ("cand_s11_f16", 1.5), ("cand_s12_f16", -1.5),
    ):
        _dump(tmp_path / arm, [1.0, 2.0, 3.0 + off, 4.0, 5.0, 6.0])
    return tmp_path


def test_families_group_the_training_seeds_of_one_recipe(results: Path) -> None:
    families = report.seed_families(results)
    assert set(families) == set(ROSTER)
    assert sorted(families["ctrl"]) == [7, 11, 12]


def test_a_partly_finished_sweep_reads_as_fewer_seeds_not_as_absent(
    results: Path,
) -> None:
    """A sweep with one job still queued must not vanish from the table."""
    (results / "cand_s12_f16" / "kdki-mean.csv").unlink()
    spread, _ = report.seed_report(results)
    assert spread.loc["cand", "n_seeds"] == 2


def test_the_family_table_carries_a_spread_not_a_point(results: Path) -> None:
    spread, _ = report.seed_report(results)
    assert spread.loc["ctrl", "n_seeds"] == 3
    assert spread.loc["ctrl", "scoring_R_sd"] > 0


def test_the_paired_row_compares_against_the_named_reference(results: Path) -> None:
    _, paired = report.seed_report(results)
    assert "cand" in paired.index
    assert REFERENCE not in paired.index, "the reference is not compared to itself"
    assert paired.loc["cand", "vs_reference"] == REFERENCE
    assert paired.loc["cand", "n_seeds"] == 3
    # cand halves the seed displacement on every seed, so it wins all three.
    assert paired.loc["cand", "scoring_n_better"] == 3


def test_no_dumps_yields_empty_tables_not_an_error(tmp_path: Path) -> None:
    spread, paired = report.seed_report(tmp_path)
    assert spread.empty
    assert paired.empty


def test_one_evaluation_set_at_a_time(
    results: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dumps for another evaluation set must not join the CASF table.

    ``results/`` holds dumps for more than one set: ``_f16`` is CASF-2016 (285
    complexes, 57 clusters) and ``_pbhold`` is the held-out PDBbind slice (281
    complexes, no clusters). Pooled, an R computed on one set lands in the same
    column as an R computed on the other, with only ``n`` and a NaN ranking to
    betray it -- and the baselines were never run on the second set at all.

    This happened: the per-run table listed three ``_pbhold`` rows at n=281
    beside CASF rows at n=285.
    """
    _dump(results / "ctrl_pbhold", [9.0, 1.0, 5.0, 2.0, 8.0, 3.0])
    monkeypatch.setattr(report, "BASELINES", ())

    casf = report.methods(results)
    assert all("pbhold" not in k for k in casf), casf
    assert any("ctrl" in k for k in casf), "CASF arms must still be found"

    other = report.methods(results, protocol="_pbhold")
    assert any("pbhold" in k for k in other), other
    assert all("_f16" not in k for k in other)


def test_each_family_is_compared_to_its_own_control(
    results: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A family that differs from ``cand`` by one thing must be paired to ``cand``.

    Comparing it to ``ctrl`` instead would credit it with everything ``cand``
    already changed. Two of the live families need this: a listwise loss applied
    on top of the union corpus, and the union corpus with complexes removed.
    """
    for arm, off in (
        ("cand2_f16", 0.4),
        ("cand2_s11_f16", 1.2),
        ("cand2_s12_f16", -1.2),
    ):
        _dump(results / arm, [1.0, 2.0, 3.0 + off, 4.0, 5.0, 6.0])
    roster = {**ROSTER, "cand2": {7: "cand2", 11: "cand2_s11", 12: "cand2_s12"}}
    monkeypatch.setattr(report, "SEED_FAMILIES", roster)
    monkeypatch.setattr(report, "SEED_REFERENCE_FOR", {"cand2": "cand"})

    _, paired = report.seed_report(results)
    assert paired.loc["cand2", "vs_reference"] == "cand", "wrong control"
    assert paired.loc["cand", "vs_reference"] == REFERENCE

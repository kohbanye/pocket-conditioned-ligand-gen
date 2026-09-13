"""One split definition, because two corpora are compared against each other.

The ProLIT CLM corpus and the stapled baseline's corpus are both cut from the
CrossDocked manifest, and a model trained on one is compared against a model
trained on the other. If the two builders disagreed about which pockets are
training data, that comparison would measure the disagreement as much as the
tokenizer -- so both call :func:`crossdocked_pocket_split` and the properties
they rely on are pinned here.

The published corpus's assignment (1670 pockets, 83 val, seed 0) was checked
against a verbatim inlining of the original code on the real manifest before
this was extracted; that check needs 2.5M rows and does not belong in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prolit.data.holdout import crossdocked_pocket_split

_ST = ["cdonly"]


def _manifest(tmp_path: Path, rows: list[dict]) -> Path:
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    path = tmp_path / "manifest.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _rows() -> list[dict]:
    rows = []
    for i in range(100):
        rows.append(
            {
                "pair_idx": i,
                "complex_dir": f"POCKET_{i // 5}",  # 5 pairs per pocket, 20 pockets
                "receptor_pdb": f"{1000 + i}_A_rec.pdb",
                "source_type": "cdonly",
                "cdonly_fold0": "train" if i < 90 else "test",
            }
        )
    return rows


def test_val_fraction_is_taken_from_the_train_pockets(tmp_path: Path) -> None:
    got = crossdocked_pocket_split(_manifest(tmp_path, _rows()), _ST, 0.10, 0)
    # 90 train rows over pockets of 5 -> 18 fold0-train pockets, 10% held out.
    assert len(got.pocket_split) == 18
    assert sum(v == "val" for v in got.pocket_split.values()) == 1


def test_assignment_is_reproducible_and_seed_dependent(tmp_path: Path) -> None:
    mp = _manifest(tmp_path, _rows())
    a = crossdocked_pocket_split(mp, _ST, 0.25, 0)
    b = crossdocked_pocket_split(mp, _ST, 0.25, 0)
    c = crossdocked_pocket_split(mp, _ST, 0.25, 1)
    assert a.pocket_split == b.pocket_split
    assert a.pocket_split != c.pocket_split


def test_a_pocket_gets_one_label_for_all_its_pairs(tmp_path: Path) -> None:
    """The property a partitioned build depends on: no pocket straddles splits."""
    got = crossdocked_pocket_split(_manifest(tmp_path, _rows()), _ST, 0.25, 0)
    by_pocket: dict[str, set[str]] = {}
    for pair, pocket in got.pair_to_pocket.items():
        by_pocket.setdefault(pocket, set()).add(got.split_of_pair(pair) or "?")
    assert all(len(v) == 1 for v in by_pocket.values())


def test_casf_receptors_are_dropped_before_anything_else(tmp_path: Path) -> None:
    got = crossdocked_pocket_split(
        _manifest(tmp_path, _rows()), _ST, 0.0, 0, casf_ids={"1000", "1001"}
    )
    # Those two pairs are gone, and so is nothing else in their pocket.
    assert 0 not in got.pair_to_pocket
    assert 1 not in got.pair_to_pocket
    assert 2 in got.pair_to_pocket


def test_excluded_pockets_leave_the_corpus_entirely(tmp_path: Path) -> None:
    got = crossdocked_pocket_split(
        _manifest(tmp_path, _rows()), _ST, 0.0, 0, exclude_pockets={"POCKET_3"}
    )
    assert "POCKET_3" not in got.pocket_split
    # And its pairs resolve to no split rather than silently to train.
    assert got.split_of_pair(15) is None


def test_a_test_fold_pocket_is_not_in_the_corpus(tmp_path: Path) -> None:
    got = crossdocked_pocket_split(_manifest(tmp_path, _rows()), _ST, 0.0, 0)
    assert got.split_of_pair(95) is None

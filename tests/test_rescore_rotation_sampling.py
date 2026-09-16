"""Dataset-level rotation augmentation: one complex per index, one frame per draw.

The corpus built with ``--num-rotations R`` holds R consecutive documents per
complex. Training on all of them is what ``e250_rot8`` did -- R times the
documents, and the encoder memorised them inside one epoch. Here the index is
one entry per COMPLEX and each access draws one of its R orientations, so an
epoch is the same size as the 1-rotation corpus while the frame varies across
epochs. These tests fix that behaviour and the checks that make it safe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from prolit.data.rescore_dataset import RescoreDataset
from prolit.tokenizers.lm_vocab import (
    BOS_ID,
    EOS_ID,
    L_CLOSE_ID,
    L_OPEN_ID,
    NUM_SPECIAL,
    P_CLOSE_ID,
    P_OPEN_ID,
)

if TYPE_CHECKING:
    from pathlib import Path

ROTATIONS = 4
COMPLEXES = 3


def _doc(protein: int, ligand: int) -> list[int]:
    return [
        BOS_ID,
        P_OPEN_ID,
        NUM_SPECIAL + protein,
        P_CLOSE_ID,
        L_OPEN_ID,
        NUM_SPECIAL + ligand,
        L_CLOSE_ID,
        EOS_ID,
    ]


def _write(
    tmp_path: Path,
    *,
    rotations: int = ROTATIONS,
    labels: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> Path:
    """A corpus of COMPLEXES complexes x ``rotations`` orientations."""
    tokens: list[int] = []
    lengths: list[int] = []
    for c in range(COMPLEXES):
        for r in range(rotations):
            # Orientation changes the codes, not the complex.
            doc = _doc(10 * c + r, 100 + 10 * c + r)
            tokens.extend(doc)
            lengths.append(len(doc))
    lab = np.repeat(np.arange(1, COMPLEXES + 1, dtype=np.float32), rotations)
    grp = np.repeat(np.arange(COMPLEXES, dtype=np.int32), rotations)
    label_stream = lab if labels is None else labels
    group_stream = grp if groups is None else groups
    (tmp_path / "train.bin").write_bytes(np.asarray(tokens, dtype=np.uint16).tobytes())
    (tmp_path / "train.len").write_bytes(np.asarray(lengths, dtype=np.uint16).tobytes())
    (tmp_path / "train.rmsd").write_bytes(label_stream.astype(np.float32).tobytes())
    (tmp_path / "train.grp").write_bytes(group_stream.astype(np.int32).tobytes())
    assert len(lengths) == COMPLEXES * rotations
    return tmp_path


def _dataset(tmp_path: Path, **kw: object) -> RescoreDataset:
    return RescoreDataset(
        tmp_path / "train.bin",
        tmp_path / "train.len",
        tmp_path / "train.rmsd",
        block_size=64,
        group_path=tmp_path / "train.grp",
        **kw,  # ty: ignore[invalid-argument-type]
    )


def test_index_is_one_entry_per_complex(tmp_path: Path) -> None:
    _write(tmp_path)
    assert len(_dataset(tmp_path, rotations_per_complex=ROTATIONS)) == COMPLEXES
    # Without the option the same corpus is served document by document.
    assert len(_dataset(tmp_path)) == COMPLEXES * ROTATIONS


def test_draws_vary_the_frame_but_not_the_label(tmp_path: Path) -> None:
    _write(tmp_path)
    ds = _dataset(tmp_path, rotations_per_complex=ROTATIONS, seed=7)
    seen: set[tuple[int, ...]] = set()
    for _ in range(40):
        item = ds[0]
        seen.add(tuple(item["input_ids"].tolist()))
        assert float(item["rmsd"]) == pytest.approx(1.0)
        assert int(item["group"]) == 0
    # All four orientations of complex 0 should appear, and nothing else.
    assert len(seen) == ROTATIONS


def test_label_must_be_constant_within_a_rotation_block(tmp_path: Path) -> None:
    bad = np.arange(COMPLEXES * ROTATIONS, dtype=np.float32)  # every doc differs
    _write(tmp_path, labels=bad)
    with pytest.raises(ValueError, match="more than one label"):
        _dataset(tmp_path, rotations_per_complex=ROTATIONS)


def test_group_must_be_constant_within_a_rotation_block(tmp_path: Path) -> None:
    bad = np.arange(COMPLEXES * ROTATIONS, dtype=np.int32)
    _write(tmp_path, groups=bad)
    with pytest.raises(ValueError, match="more than one group"):
        _dataset(tmp_path, rotations_per_complex=ROTATIONS)


def test_doc_count_must_be_a_multiple_of_the_rotations(tmp_path: Path) -> None:
    _write(tmp_path)
    with pytest.raises(ValueError, match="not a multiple"):
        _dataset(tmp_path, rotations_per_complex=ROTATIONS + 1)


def test_refuses_to_compose_with_per_doc_selection(tmp_path: Path) -> None:
    _write(tmp_path)
    with pytest.raises(ValueError, match="does not compose"):
        _dataset(tmp_path, rotations_per_complex=ROTATIONS, max_docs=4)
    with pytest.raises(ValueError, match="does not compose"):
        _dataset(tmp_path, rotations_per_complex=ROTATIONS, divide_by_size=True)

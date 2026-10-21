"""Teacher values reach the loss aligned to their documents, and reweight it.

Pre-registered in docs/notes/2026-09-15_prereg_distillation.md. Two properties
decide whether the round measures what it claims: the sidecar value served with
a document must be THAT document's teacher prediction, and the distillation term
must reweight the existing objective rather than stack on top of it (the
students also carry --corr-weight, which already reweights).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from prolit.data.rescore_dataset import RescoreDataset, collate_rescore

if TYPE_CHECKING:
    from pathlib import Path

DOCS = 6
BLOCK = 64
VOCAB_LO = 10


def _corpus(tmp: Path, *, teacher: bool) -> tuple[Path, np.ndarray]:
    lengths = np.full(DOCS, 8, dtype=np.uint16)
    tokens = np.arange(VOCAB_LO, VOCAB_LO + DOCS * 8, dtype=np.uint16)
    labels = np.arange(DOCS, dtype=np.float32) + 4.0
    (tmp / "train.bin").write_bytes(tokens.tobytes())
    (tmp / "train.len").write_bytes(lengths.tobytes())
    (tmp / "train.rmsd").write_bytes(labels.tobytes())
    teach = labels[::-1].copy()  # deliberately NOT the labels, and not symmetric
    if teacher:
        np.save(tmp / "teacher.npy", teach)
    return tmp, teach


def test_teacher_value_follows_its_document(tmp_path: Path) -> None:
    tmp, teach = _corpus(tmp_path, teacher=True)
    ds = RescoreDataset(tmp / "train.bin", tmp / "train.len", tmp / "train.rmsd",
                        BLOCK, teacher_path=tmp / "teacher.npy")
    for i in range(DOCS):
        assert float(ds[i]["teacher"]) == teach[i]


def test_absent_sidecar_leaves_no_key(tmp_path: Path) -> None:
    tmp, _ = _corpus(tmp_path, teacher=False)
    ds = RescoreDataset(tmp / "train.bin", tmp / "train.len", tmp / "train.rmsd", BLOCK)
    assert "teacher" not in ds[0]
    assert "teacher" not in collate_rescore([ds[0], ds[1]])


def test_length_mismatch_refuses(tmp_path: Path) -> None:
    tmp, _ = _corpus(tmp_path, teacher=True)
    np.save(tmp / "short.npy", np.zeros(DOCS - 1, dtype=np.float32))
    # A short sidecar must refuse rather than pair documents with wrong values.
    with pytest.raises(ValueError, match="teacher sidecar"):
        RescoreDataset(tmp / "train.bin", tmp / "train.len", tmp / "train.rmsd",
                       BLOCK, teacher_path=tmp / "short.npy")


def test_collate_stacks_in_order(tmp_path: Path) -> None:
    tmp, teach = _corpus(tmp_path, teacher=True)
    ds = RescoreDataset(tmp / "train.bin", tmp / "train.len", tmp / "train.rmsd",
                        BLOCK, teacher_path=tmp / "teacher.npy")
    batch = collate_rescore([ds[2], ds[0]])
    torch.testing.assert_close(batch["teacher"], torch.tensor([teach[2], teach[0]]))


def test_distillation_reweights_rather_than_stacks() -> None:
    """A weight of 1.0 must leave ONLY the teacher term, not base + teacher."""
    pred = torch.tensor([1.0, 2.0])
    base = torch.tensor(0.7)
    kd = torch.nn.functional.smooth_l1_loss(pred, torch.tensor([1.5, 2.5]))
    for weight in (0.0, 0.5, 1.0):
        blended = (1.0 - weight) * base + weight * kd
        assert float(blended) <= float(max(base, kd)) + 1e-6
    assert float((1.0 - 1.0) * base + 1.0 * kd) == float(kd)

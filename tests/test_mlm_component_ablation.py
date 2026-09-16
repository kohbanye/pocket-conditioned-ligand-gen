"""Loss weighting and masking-topology contracts for affinity MLM ablations."""

import socket
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from prolit.data.mlm_dataset import MLMTokenDataset
from prolit.model.mlm import component_balanced_mlm_loss
from prolit.tokenizers.lm_vocab import L_CLOSE_ID, L_OPEN_ID, P_CLOSE_ID, P_OPEN_ID


def test_component_loss_equalizes_unequal_lengths_and_handles_empty() -> None:
    ids = torch.tensor([[P_OPEN_ID, 7, 8, 9, P_CLOSE_ID, L_OPEN_ID, 10, L_CLOSE_ID]])
    labels = torch.tensor([[-100, 7, 8, 9, -100, -100, 10, -100]])
    logits = torch.randn(1, 8, 16, requires_grad=True)
    losses = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none")
    expected = (losses[1:4].mean() + losses[6]) / 2
    actual = component_balanced_mlm_loss(logits, labels, ids)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    ligand_labels = labels.clone()
    ligand_labels[:, 1:4] = -100
    torch.testing.assert_close(
        component_balanced_mlm_loss(logits, ligand_labels, ids), losses[6]
    )
    empty = component_balanced_mlm_loss(logits, torch.full_like(labels, -100), ids)
    assert empty.item() == 0
    assert empty.requires_grad


def test_block_mask_preserves_component_counts_and_specials(tmp_path: Path) -> None:
    doc = [P_OPEN_ID, *range(7, 27), P_CLOSE_ID, L_OPEN_ID, *range(27, 47), L_CLOSE_ID]
    np.array(doc, dtype=np.uint16).tofile(tmp_path / "train.bin")
    np.array([len(doc)], dtype=np.uint16).tofile(tmp_path / "train.len")
    options = {
        "block_size": 64,
        "base_vocab_size": 64,
        "mask_token_id": 64,
        "mask_prob": 0.5,
    }
    random = MLMTokenDataset(tmp_path / "train.bin", tmp_path / "train.len", **options)
    block = MLMTokenDataset(
        tmp_path / "train.bin",
        tmp_path / "train.len",
        ligand_block_masking=True,
        **options,
    )
    for epoch in range(12):
        random.set_epoch(epoch)
        block.set_epoch(epoch)
        a, b = random[0], block[0]
        am, bm = a["labels"] != -100, b["labels"] != -100
        assert am.sum() == bm.sum()
        assert am[1:21].sum() == bm[1:21].sum()
        assert am[23:43].sum() == bm[23:43].sum()
        selected = torch.where(bm[23:43])[0]
        assert (selected.diff() == 1).all()
        for i in (0, 21, 22, 43):
            assert b["input_ids"][i] == doc[i]
            assert not bm[i]


def test_epoch_updates_reach_persistent_workers(tmp_path: Path) -> None:
    # PyTorch tensor transport needs a local socket; the agent sandbox may
    # disallow setsockopt even though worker processes themselves can launch.
    try:
        with socket.socket(socket.AF_UNIX) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except PermissionError:
        pytest.skip("sandbox blocks local socket transport needed by DataLoader")
    doc = [P_OPEN_ID, *range(7, 27), P_CLOSE_ID, L_OPEN_ID, *range(27, 47), L_CLOSE_ID]
    np.array(doc, dtype=np.uint16).tofile(tmp_path / "train.bin")
    np.array([len(doc)], dtype=np.uint16).tofile(tmp_path / "train.len")
    ds = MLMTokenDataset(
        tmp_path / "train.bin",
        tmp_path / "train.len",
        block_size=64,
        base_vocab_size=64,
        mask_token_id=64,
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        num_workers=1,
        persistent_workers=True,
        timeout=120,
        multiprocessing_context="spawn",
    )
    first = next(iter(loader))["labels"][0]
    ds.set_epoch(1)
    second = next(iter(loader))["labels"][0]
    assert not torch.equal(first, second)
    torch.testing.assert_close(second, ds[0]["labels"])
    ds.set_epoch(0)
    torch.testing.assert_close(next(iter(loader))["labels"][0], first)

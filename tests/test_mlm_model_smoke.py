"""Smoke test: tiny ESM3-style MLM builds, runs forward/backward over masks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from src.config import ComplexMLMConfig
from src.data.mlm_dataset import MLMTokenDataset, collate_mlm
from src.model.complex_mlm import build_complex_mlm, count_parameters
from src.tokenizers.lm_vocab import (
    BOS_ID,
    EOS_ID,
    L_CLOSE_ID,
    L_OPEN_ID,
    P_CLOSE_ID,
    P_OPEN_ID,
)

if TYPE_CHECKING:
    from pathlib import Path

_DOC = [
    BOS_ID,
    P_OPEN_ID,
    10,
    11,
    12,
    P_CLOSE_ID,
    L_OPEN_ID,
    20,
    21,
    22,
    L_CLOSE_ID,
    EOS_ID,
]


def _tiny_config() -> ComplexMLMConfig:
    return ComplexMLMConfig(
        atom_codebook_size=32,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
    )


def test_esm_mlm_forward_backward(tmp_path: Path) -> None:
    cfg = _tiny_config()
    model = build_complex_mlm(cfg)
    model.train()

    # A tiny 3-doc cache -> masked batch through the real collate path.
    flat = np.concatenate([np.asarray(_DOC, dtype=np.uint16)] * 3)
    lengths = np.asarray([len(_DOC)] * 3, dtype=np.uint16)
    (tmp_path / "train.bin").write_bytes(flat.tobytes())
    lengths.tofile(tmp_path / "train.len")
    ds = MLMTokenDataset(
        tmp_path / "train.bin",
        tmp_path / "train.len",
        block_size=64,
        base_vocab_size=cfg.base_vocab_size,
        mask_token_id=cfg.mask_token_id,
        mask_prob=0.5,
    )
    batch = collate_mlm([ds[0], ds[1], ds[2]])

    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    # Logits over the FULL vocab (base + <mask>); loss finite over masked tokens.
    assert out.logits.shape == (3, len(_DOC), cfg.vocab_size)
    assert torch.isfinite(out.loss)
    assert out.loss.item() > 0

    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_mask_token_id_is_embeddable() -> None:
    """The appended <mask> id must be a valid embedding row (no off-by-one)."""
    cfg = _tiny_config()
    model = build_complex_mlm(cfg)
    emb = model.get_input_embeddings()
    assert emb.num_embeddings == cfg.vocab_size
    ids = torch.tensor([[cfg.mask_token_id, L_OPEN_ID, 7]])
    # Must not raise an index error.
    model(input_ids=ids, attention_mask=torch.ones_like(ids))


def test_padded_batch_produces_finite_loss(tmp_path: Path) -> None:
    """Key-padding bias must not create NaNs (guards 0 * -inf)."""
    cfg = _tiny_config()
    model = build_complex_mlm(cfg)
    short = [BOS_ID, L_OPEN_ID, 20, 21, L_CLOSE_ID, EOS_ID]
    flat = np.concatenate(
        [np.asarray(_DOC, dtype=np.uint16), np.asarray(short, dtype=np.uint16)]
    )
    lengths = np.asarray([len(_DOC), len(short)], dtype=np.uint16)
    (tmp_path / "train.bin").write_bytes(flat.tobytes())
    lengths.tofile(tmp_path / "train.len")
    ds = MLMTokenDataset(
        tmp_path / "train.bin",
        tmp_path / "train.len",
        block_size=64,
        base_vocab_size=cfg.base_vocab_size,
        mask_token_id=cfg.mask_token_id,
        mask_prob=0.5,
    )
    batch = collate_mlm([ds[0], ds[1]])  # ragged -> padding present
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    assert torch.isfinite(out.logits).all()
    assert torch.isfinite(out.loss)


def test_default_config_is_about_100m() -> None:
    """Default architecture should land near the ~100M target."""
    model = build_complex_mlm(ComplexMLMConfig(atom_codebook_size=8192))
    n = count_parameters(model)
    assert 80e6 < n < 120e6, f"{n / 1e6:.1f}M outside [80, 120]M"

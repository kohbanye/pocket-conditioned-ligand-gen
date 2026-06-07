"""LightningModule for from-scratch training of the pocket-conditioned LM."""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
import torch
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.data.lm_dataset import PAD_SEGMENT
from src.model.ligand_lm import build_qwen3_lm, count_parameters

if TYPE_CHECKING:
    from src.config import LMTrainingConfig


def build_block_diagonal_mask(segment_ids: Tensor, dtype: torch.dtype) -> Tensor:
    """Build a 4D additive attention mask from packed-document segment ids.

    A query at position ``i`` may attend key ``j`` iff ``j <= i`` (causal) and
    both belong to the same document (``segment_ids[i] == segment_ids[j]``).
    Right-padding shares ``PAD_SEGMENT`` so padded queries still attend earlier
    padding (and themselves), avoiding fully-masked rows / NaNs.

    Args:
        segment_ids: ``(B, L)`` long; ``PAD_SEGMENT`` (-1) on padding.
        dtype: floating dtype for the additive mask.

    Returns:
        ``(B, 1, L, L)`` additive mask: ``0`` where attention is allowed and
        the dtype's most-negative value where it is masked.
    """
    b, length = segment_ids.shape
    device = segment_ids.device
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
    same = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)  # (B, L, L)
    allowed = same & causal.unsqueeze(0)
    mask = torch.zeros(b, 1, length, length, dtype=dtype, device=device)
    mask.masked_fill_(~allowed.unsqueeze(1), torch.finfo(dtype).min)
    return mask


class LigandLMModule(L.LightningModule):
    """Trains a dense Qwen3 LM on packed VQ-VAE token blocks (loss on all tokens)."""

    def __init__(self, config: LMTrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.model = build_qwen3_lm(config.model)
        self._logged_param_count = False

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        # Build the block-diagonal causal mask in the parameter dtype so it
        # matches the attention scores' dtype under autocast.
        attn_dtype = self.model.get_input_embeddings().weight.dtype
        attn_mask = build_block_diagonal_mask(batch["segment_ids"], attn_dtype)
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=attn_mask,
            position_ids=batch["position_ids"],
            labels=batch["labels"],
        )
        return out.loss

    def on_fit_start(self) -> None:
        if not self._logged_param_count:
            n = count_parameters(self.model)
            self.print(f"Model parameters: {n / 1e6:.1f}M")
            self._logged_param_count = True

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:  # noqa: ARG002
        loss = self(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/ppl", torch.exp(loss.detach()))
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:  # noqa: ARG002
        loss = self(batch)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/ppl", torch.exp(loss.detach()), sync_dist=True)

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:  # noqa: ARG002
        loss = self(batch)
        self.log("test/loss", loss, sync_dist=True)
        self.log("test/ppl", torch.exp(loss.detach()), sync_dist=True)

    def configure_optimizers(self) -> dict:
        # Decoupled weight decay: skip 1D params (norms, biases). Embeddings are
        # tied to the LM head, so decaying them is standard practice here.
        decay, no_decay = [], []
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            (decay if param.ndim >= 2 else no_decay).append(param)  # noqa: PLR2004
        opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
        )

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, min(self.config.warmup_steps, total_steps - 1))
        cosine_steps = max(1, total_steps - warmup_steps)
        warmup = LinearLR(
            opt,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            opt,
            T_max=cosine_steps,
            eta_min=self.config.learning_rate * self.config.min_lr_ratio,
        )
        scheduler = SequentialLR(
            opt,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# Re-export so callers can build masks without importing the dataset module.
__all__ = ["PAD_SEGMENT", "LigandLMModule", "build_block_diagonal_mask"]

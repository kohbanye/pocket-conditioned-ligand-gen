"""LightningModule for from-scratch training of the complex-token MLM.

Wraps the ESM masked-LM (:func:`prolit.model.complex_mlm.build_esm_mlm`) with a
BERT-style masked-token objective. Loss and metrics are computed over the
masked positions only (``labels != IGNORE_INDEX``); the rest of the sequence is
context. The optimizer/schedule mirrors :class:`~prolit.model.lm_module.LigandLMModule`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
import torch
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from prolit.data.mlm_dataset import IGNORE_INDEX
from prolit.model.complex_mlm import build_complex_mlm, count_parameters

if TYPE_CHECKING:
    from prolit.config import MLMTrainingConfig


class ComplexMLMModule(L.LightningModule):
    """Trains a bidirectional ESM MLM on per-complex VQ-VAE token sequences."""

    def __init__(self, config: MLMTrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.model = build_complex_mlm(config.model)
        self._logged_param_count = False

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return out.loss, out.logits

    def on_fit_start(self) -> None:
        if not self._logged_param_count:
            n = count_parameters(self.model)
            self.print(f"Model parameters: {n / 1e6:.1f}M")
            self._logged_param_count = True

    def _masked_accuracy(self, logits: Tensor, labels: Tensor) -> Tensor | None:
        mask = labels != IGNORE_INDEX
        if not bool(mask.any()):
            return None
        preds = logits.argmax(dim=-1)
        return (preds[mask] == labels[mask]).float().mean()

    def _step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        loss, logits = self(batch)
        sync = stage != "train"
        self.log(f"{stage}/loss", loss, prog_bar=stage != "test", sync_dist=sync)
        self.log(f"{stage}/ppl", torch.exp(loss.detach()), sync_dist=sync)
        acc = self._masked_accuracy(logits.detach(), batch["labels"])
        if acc is not None:
            self.log(f"{stage}/acc", acc, prog_bar=stage == "train", sync_dist=sync)
        return loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:  # noqa: ARG002
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:  # noqa: ARG002
        self._step(batch, "val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:  # noqa: ARG002
        self._step(batch, "test")

    def configure_optimizers(self) -> dict:
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
            opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = CosineAnnealingLR(
            opt,
            T_max=cosine_steps,
            eta_min=self.config.learning_rate * self.config.min_lr_ratio,
        )
        scheduler = SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


__all__ = ["ComplexMLMModule"]

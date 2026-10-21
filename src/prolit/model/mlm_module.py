"""LightningModule for from-scratch training of the complex-token MLM.

Wraps the ESM-style masked LM (:func:`prolit.model.mlm.build_prolit_mlm`) with a
BERT-style masked-token objective. Loss and metrics are computed over the
masked positions only (``labels != IGNORE_INDEX``); the rest of the sequence is
context. The optimizer/schedule mirrors
:class:`~prolit.model.clm_module.ProLITCLMModule`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
import torch
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from prolit.data.mlm_dataset import IGNORE_INDEX
from prolit.model.mlm import MLMOutput, build_prolit_mlm, count_parameters

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

    from prolit.config import MLMTrainingConfig


class ProLITMLMModule(L.LightningModule):
    """Trains a bidirectional ESM MLM on per-complex VQ-VAE token sequences."""

    def __init__(self, config: MLMTrainingConfig) -> None:
        super().__init__()
        self.config: MLMTrainingConfig = config
        self.save_hyperparameters()
        self.model = build_prolit_mlm(config.model)
        self._logged_param_count: bool = False
        if config.freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self) -> None:
        """Everything but the auxiliary heads: ~99M parameters held fixed.

        Put in eval mode as well as frozen. A frozen module still applies
        dropout in train mode, which would hand the heads a different vector
        for the same complex on every step -- noise they cannot fit and are not
        meant to. Mirrors ``ComplexRescoreModule._freeze_encoder``, including
        the :meth:`train` override that keeps it that way.
        """
        for name, param in self.model.named_parameters():
            param.requires_grad = name.startswith("contact_heads.")
        for name, child in self.model.named_children():
            if name != "contact_heads":
                child.eval()

    def train(self, mode: bool = True) -> ProLITMLMModule:  # noqa: FBT001, FBT002
        """Lightning calls this each epoch and ``nn.Module.train`` recurses.

        Without the override the encoder is put back into train mode at the
        start of every epoch and starts applying dropout again: frozen weights,
        but a moving output.
        """
        super().train(mode)
        if self.config.freeze_encoder:
            for name, child in self.model.named_children():
                if name != "contact_heads":
                    child.eval()
        return self

    def forward(self, batch: dict[str, Tensor]) -> MLMOutput:
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            component_balanced_loss=getattr(
                self.config, "component_balanced_loss", False
            ),
            contact_targets={
                name: batch[f"contact_{name}"]
                for name in ("res", "itype")
                if f"contact_{name}" in batch
            },
        )

    def on_fit_start(self) -> None:
        if not self._logged_param_count:
            n = count_parameters(self.model)
            self.print(f"Model parameters: {n / 1e6:.1f}M")
            self._logged_param_count: bool = True

    def on_train_epoch_start(self) -> None:
        # Validation retains its fixed mask schedule for comparable checkpoints.
        # Lightning attaches `datamodule` to the trainer at fit time; not in the stub.
        dm = self.trainer.datamodule  # ty: ignore[unresolved-attribute]
        datasets = getattr(dm, "_datasets", {})
        if "train" in datasets:
            datasets["train"].set_epoch(self.current_epoch)

    def _masked_accuracy(self, logits: Tensor, labels: Tensor) -> Tensor | None:
        mask = labels != IGNORE_INDEX
        if not bool(mask.any()):
            return None
        preds = logits.argmax(dim=-1)
        return (preds[mask] == labels[mask]).float().mean()

    def _step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        out = self(batch)
        loss, logits = out.loss, out.logits
        sync = stage != "train"
        # Logged before the auxiliary terms are added, so this column stays
        # comparable with every checkpoint trained before the heads existed.
        self.log(f"{stage}/loss", loss, prog_bar=stage != "test", sync_dist=sync)
        self.log(f"{stage}/ppl", torch.exp(loss.detach()), sync_dist=sync)
        acc = self._masked_accuracy(logits.detach(), batch["labels"])
        if acc is not None:
            self.log(f"{stage}/acc", acc, prog_bar=stage == "train", sync_dist=sync)

        total = loss
        weights = getattr(self.config, "contact_loss_weights", {}) or {}
        for name, aux in out.contact_losses.items():
            w = float(weights.get(name, 0.0))
            self.log(f"{stage}/contact_{name}", aux.detach(), sync_dist=sync)
            aux_acc = self._masked_accuracy(
                out.contact_logits[name].detach(), batch[f"contact_{name}"]
            )
            if aux_acc is not None:
                self.log(f"{stage}/contact_{name}_acc", aux_acc, sync_dist=sync)
            if w:
                total = total + w * aux
        if total is not loss:
            self.log(f"{stage}/total", total.detach(), sync_dist=sync)
        return total

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:  # noqa: ARG002
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:  # noqa: ARG002
        self._step(batch, "val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:  # noqa: ARG002
        self._step(batch, "test")

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
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


__all__ = ["ProLITMLMModule"]

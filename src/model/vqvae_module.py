"""Lightning module for joint protein + ligand VQ-VAE training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
import torch
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.tokenizers.vqvae import TransformerVQVAE

if TYPE_CHECKING:
    from src.config import (
        AtomVQVAETrainingConfig,
    )


class AtomVQVAEModule(L.LightningModule):
    """Unified all-atom VQ-VAE: one codebook over protein + ligand atoms.

    Consumes a single ``(x, mask)`` batch of atom-descriptor sequences (a
    protein pocket OR a ligand; the ``source`` slot disambiguates). Per-source
    loss masking (aa/bb_sc -> protein rows, clash -> ligand rows) lives in
    :meth:`TransformerVQVAE._compute_recon_loss`.
    """

    def __init__(self, config: AtomVQVAETrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.vqvae = TransformerVQVAE(config.atom)
        self.automatic_optimization = False

    def on_fit_start(self) -> None:
        self._inject_normalization_stats()

    def on_validation_start(self) -> None:
        self._inject_normalization_stats()

    def on_test_start(self) -> None:
        self._inject_normalization_stats()

    def _inject_normalization_stats(self) -> None:
        dm = getattr(self.trainer, "datamodule", None)
        stats = getattr(dm, "norm_stats", None) if dm is not None else None
        if stats is not None and "atom_mean" in stats and "atom_std" in stats:
            self.vqvae.set_normalization(stats["atom_mean"], stats["atom_std"])

    @staticmethod
    def _combine_losses(
        out: dict[str, Tensor],
        recon_weights: dict[str, float],
    ) -> Tensor:
        head_losses: dict[str, Tensor] = out["head_losses"]
        commit = out["commitment_loss"]
        weighted = sum(
            recon_weights.get(name, 1.0) * loss for name, loss in head_losses.items()
        )
        return weighted + commit

    def _run(
        self,
        prefix: str,
        batch: tuple[Tensor, Tensor],
        *,
        train: bool,
    ) -> Tensor:
        x, mask = batch
        out = self.vqvae(x, mask=mask)
        loss = self._combine_losses(out, self.config.atom.recon_weights)
        self.log(f"{prefix}_total", loss, sync_dist=not train)
        self.log(f"{prefix}_commit", out["commitment_loss"], sync_dist=not train)
        for name, hl in out["head_losses"].items():
            self.log(f"{prefix}_{name}", hl, sync_dist=not train)
        self._log_utilization(
            prefix, out["indices"][mask], self.config.atom.codebook_size
        )
        for key, value in out["diagnostics"].items():
            self.log(f"{prefix}_{key}", value.float(), sync_dist=not train)
        return loss

    def _log_utilization(self, prefix: str, indices: Tensor, size: int) -> None:
        if indices.numel() == 0:
            return
        self.log(f"{prefix}_codebook_util", indices.unique().numel() / size)
        counts = torch.bincount(indices, minlength=size).float()
        probs = counts / counts.sum().clamp_min(1.0)
        probs = probs[probs > 0]
        self.log(f"{prefix}_perplexity", (-(probs * probs.log()).sum()).exp())

    def training_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        opt = self.optimizers()
        if not isinstance(opt, torch.optim.Optimizer):  # pragma: no cover
            msg = "Expected a single optimizer"
            raise TypeError(msg)
        sch = self.lr_schedulers()

        loss = self._run("train/atom", batch, train=True)
        self.log("train/total_loss", loss, prog_bar=True)

        opt.zero_grad()
        self.manual_backward(loss)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.log("train/grad_norm", grad_norm)
        opt.step()
        if sch is not None:
            sch.step()
            self.log("train/lr", opt.param_groups[0]["lr"])

    def validation_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        self._run("val/atom", batch, train=False)

    def test_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        self._run("test/atom", batch, train=False)

    def configure_optimizers(self) -> dict:
        opt = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, min(500, total_steps // 20))
        cosine_steps = max(1, total_steps - warmup_steps)
        warmup = LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = CosineAnnealingLR(
            opt, T_max=cosine_steps, eta_min=self.config.learning_rate * 0.01
        )
        scheduler = SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

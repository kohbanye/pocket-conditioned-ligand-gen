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
        LigandVQVAEConfig,
        ProteinVQVAEConfig,
        VQVAETrainingConfig,
    )


class VQVAEModule(L.LightningModule):
    """Joint training of protein and ligand structure VQ-VAEs.

    Loss aggregation is a simple weighted sum: each VQ-VAE forward returns
    per-head losses (continuous coord MSE in Cartesian Å² + per-categorical
    CE), the module multiplies by the per-head weights from the domain
    config, and adds the commitment loss.
    """

    def __init__(self, config: VQVAETrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.protein_vqvae = TransformerVQVAE(config.protein)
        self.ligand_vqvae = TransformerVQVAE(config.ligand)

        self.automatic_optimization = False

    # ------------------------------------------------------------------
    # Lifecycle: inject descriptor normalization stats
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        self._inject_normalization_stats()

    def on_validation_start(self) -> None:
        self._inject_normalization_stats()

    def on_test_start(self) -> None:
        self._inject_normalization_stats()

    def _inject_normalization_stats(self) -> None:
        dm = getattr(self.trainer, "datamodule", None)
        stats = getattr(dm, "norm_stats", None) if dm is not None else None
        if stats is None:
            return
        if "protein_mean" in stats and "protein_std" in stats:
            self.protein_vqvae.set_normalization(
                stats["protein_mean"],
                stats["protein_std"],
            )
        if "ligand_mean" in stats and "ligand_std" in stats:
            self.ligand_vqvae.set_normalization(
                stats["ligand_mean"],
                stats["ligand_std"],
            )

    # ------------------------------------------------------------------
    # Loss combination
    # ------------------------------------------------------------------

    @staticmethod
    def _combine_losses(
        out: dict[str, Tensor],
        recon_weights: dict[str, float],
    ) -> Tensor:
        """Combine head_losses with config weights, then add commitment loss."""
        head_losses: dict[str, Tensor] = out["head_losses"]
        commit = out["commitment_loss"]
        weighted = sum(
            recon_weights.get(name, 1.0) * loss for name, loss in head_losses.items()
        )
        return weighted + commit

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def _run_branch(
        self,
        prefix: str,
        vqvae: TransformerVQVAE,
        cfg: ProteinVQVAEConfig | LigandVQVAEConfig,
        batch: tuple[Tensor, Tensor],
        *,
        log_train_only: bool,
    ) -> Tensor:
        x, mask = batch
        out = vqvae(x, mask=mask)
        loss = self._combine_losses(out, cfg.recon_weights)

        # Always log scalar losses + per-head losses for triage.
        self.log(f"{prefix}_total", loss, sync_dist=not log_train_only)
        self.log(
            f"{prefix}_commit",
            out["commitment_loss"],
            sync_dist=not log_train_only,
        )
        for name, hl in out["head_losses"].items():
            self.log(f"{prefix}_{name}", hl, sync_dist=not log_train_only)

        # Codebook utilization on real tokens only.
        real_indices = out["indices"][mask]
        self._log_utilization(prefix, real_indices, cfg.codebook_size)
        if log_train_only:
            self._log_diagnostics(prefix, out["diagnostics"])
        else:
            self._log_diagnostics(prefix, out["diagnostics"], sync_dist=True)
        return loss

    def training_step(
        self,
        batch: dict[str, tuple],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        opt = self.optimizers()
        if not isinstance(opt, torch.optim.Optimizer):  # pragma: no cover
            msg = "Expected a single optimizer"
            raise TypeError(msg)
        sch = self.lr_schedulers()

        total_loss = torch.tensor(0.0, device=self.device)

        if "protein" in batch:
            total_loss = total_loss + self._run_branch(
                "train/protein",
                self.protein_vqvae,
                self.config.protein,
                batch["protein"],
                log_train_only=True,
            )
        if "ligand" in batch:
            total_loss = total_loss + self._run_branch(
                "train/ligand",
                self.ligand_vqvae,
                self.config.ligand,
                batch["ligand"],
                log_train_only=True,
            )

        self.log("train/total_loss", total_loss, prog_bar=True)

        opt.zero_grad()
        self.manual_backward(total_loss)
        self._log_submodule_grad_norms()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.log("train/grad_norm", grad_norm)
        opt.step()
        self._log_submodule_param_norms()
        if sch is not None:
            sch.step()
            self.log("train/lr", opt.param_groups[0]["lr"])

    def validation_step(
        self,
        batch: dict[str, tuple],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        if "protein" in batch:
            self._run_branch(
                "val/protein",
                self.protein_vqvae,
                self.config.protein,
                batch["protein"],
                log_train_only=False,
            )
        if "ligand" in batch:
            self._run_branch(
                "val/ligand",
                self.ligand_vqvae,
                self.config.ligand,
                batch["ligand"],
                log_train_only=False,
            )

    def test_step(
        self,
        batch: dict[str, tuple],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        if "protein" in batch:
            self._run_branch(
                "test/protein",
                self.protein_vqvae,
                self.config.protein,
                batch["protein"],
                log_train_only=False,
            )
        if "ligand" in batch:
            self._run_branch(
                "test/ligand",
                self.ligand_vqvae,
                self.config.ligand,
                batch["ligand"],
                log_train_only=False,
            )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_utilization(
        self,
        prefix: str,
        indices: Tensor,
        codebook_size: int,
    ) -> None:
        unique_codes = indices.unique().numel()
        utilization = unique_codes / codebook_size
        self.log(f"{prefix}_codebook_util", utilization)

        counts = torch.bincount(indices, minlength=codebook_size).float()
        probs = counts / counts.sum().clamp_min(1.0)
        probs = probs[probs > 0]
        entropy = -(probs * probs.log()).sum()
        self.log(f"{prefix}_perplexity", entropy.exp())

    def _log_diagnostics(
        self,
        prefix: str,
        diagnostics: dict[str, Tensor],
        *,
        sync_dist: bool = False,
    ) -> None:
        for key, value in diagnostics.items():
            self.log(f"{prefix}_{key}", value.float(), sync_dist=sync_dist)

    def _submodules_for_stats(
        self,
    ) -> list[tuple[str, str, torch.nn.Module]]:
        return [
            ("protein", "encoder", self.protein_vqvae.transformer_encoder),
            ("protein", "decoder", self.protein_vqvae.transformer_decoder),
            ("protein", "latent_proj", self.protein_vqvae.latent_proj),
            ("ligand", "encoder", self.ligand_vqvae.transformer_encoder),
            ("ligand", "decoder", self.ligand_vqvae.transformer_decoder),
            ("ligand", "latent_proj", self.ligand_vqvae.latent_proj),
        ]

    def _log_submodule_grad_norms(self) -> None:
        for model, sub, module in self._submodules_for_stats():
            acc = torch.zeros((), device=self.device)
            for p in module.parameters():
                if p.grad is not None:
                    acc = acc + p.grad.detach().float().pow(2).sum()
            self.log(f"train/{model}_grad_norm_{sub}", acc.sqrt())

    def _log_submodule_param_norms(self) -> None:
        for model, sub, module in self._submodules_for_stats():
            acc = torch.zeros((), device=self.device)
            for p in module.parameters():
                acc = acc + p.detach().float().pow(2).sum()
            self.log(f"train/{model}_param_norm_{sub}", acc.sqrt())

    def configure_optimizers(self) -> dict:
        opt = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, min(500, total_steps // 20))
        cosine_steps = max(1, total_steps - warmup_steps)

        warmup = LinearLR(
            opt,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            opt,
            T_max=cosine_steps,
            eta_min=self.config.learning_rate * 0.01,
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
        if getattr(self.config.atom, "split_codebook", False):
            self._log_split_utilization(prefix, x, out["indices"], mask, train=train)
        else:
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

    def _log_split_utilization(
        self,
        prefix: str,
        x: Tensor,
        indices: Tensor,
        mask: Tensor,
        *,
        train: bool,  # noqa: ARG002
    ) -> None:
        # Protein and ligand indices share the 0-based value range but come from
        # different books; utilisation / perplexity must be computed per source.
        from src.tokenizers.descriptor_schema import (  # noqa: PLC0415
            ATOM_LAYOUT,
            SOURCE_LIGAND_IDX,
            SOURCE_PROTEIN_IDX,
            fields_by_name,
        )

        source = x[..., fields_by_name(ATOM_LAYOUT)["source"].start].long()
        for tag, sidx, size in (
            ("protein", SOURCE_PROTEIN_IDX, self.config.atom.codebook_size),
            ("ligand", SOURCE_LIGAND_IDX, self.config.atom.ligand_codebook_size),
        ):
            sel = mask & (source == sidx)
            self._log_utilization(f"{prefix}_{tag}", indices[sel], size)

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

"""Lightning module for joint protein + ligand VQ-VAE training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.tokenizers.vqvae import TransformerVQVAE

if TYPE_CHECKING:
    from src.config import VQVAETrainingConfig


class TaskWeighting(nn.Module):
    """Kendall & Gal (2018) uncertainty weighting for two scalar losses.

    Parameterises each task's observation noise as ``s = log σ²`` (a single
    learnable scalar). The combined objective
    ``exp(-s_r) * recon + s_r + exp(-s_c) * coord + s_c`` has a unique
    stationary point at ``exp(-s_i) = 1 / L_i``, so the weight of each task
    is automatically matched to its current loss scale — no sweep needed.

    Notes on scope:
        * Only used when ``coord_loss_enabled`` is True. When disabled, the
          caller should fall back to a plain ``recon + commitment`` sum so
          ``s_c`` does not drift to -∞ under identically-zero ``coord``.
        * The commitment loss is *not* routed through this module: its
          weight is tied to codebook semantics (not task balance).
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_var_recon = nn.Parameter(torch.zeros(()))
        self.log_var_coord = nn.Parameter(torch.zeros(()))

    def forward(self, recon: Tensor, coord: Tensor) -> Tensor:
        return (
            torch.exp(-self.log_var_recon) * recon
            + self.log_var_recon
            + torch.exp(-self.log_var_coord) * coord
            + self.log_var_coord
        )


class VQVAEModule(L.LightningModule):
    """Joint training of protein and ligand structure VQ-VAEs.

    Trains both VQ-VAEs simultaneously with combined loss.
    Logs per-model reconstruction loss, commitment loss, and
    codebook utilization metrics.
    """

    def __init__(self, config: VQVAETrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.protein_vqvae = TransformerVQVAE(config.protein)
        self.ligand_vqvae = TransformerVQVAE(config.ligand)

        # Per-model uncertainty weighting (only active when coord loss is on).
        self.protein_task_weighting = TaskWeighting()
        self.ligand_task_weighting = TaskWeighting()

        # Track codebook usage for utilization metrics
        self.automatic_optimization = False

    # ------------------------------------------------------------------
    # Lifecycle: inject descriptor normalization stats into each VQ-VAE
    # so the coord loss can denormalize descriptors back to Å.
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        self._inject_normalization_stats()

    def on_validation_start(self) -> None:
        self._inject_normalization_stats()

    def on_test_start(self) -> None:
        self._inject_normalization_stats()

    def _inject_normalization_stats(self) -> None:
        """Push ``norm_stats`` from the DataModule into each VQ-VAE buffer."""
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
    # Batch unpacking helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack(
        entry: tuple,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Accept (x, mask) or (x, aux, mask). Returns (x, mask, aux)."""
        if len(entry) == 2:  # noqa: PLR2004
            x, mask = entry
            return x, mask, None
        if len(entry) == 3:  # noqa: PLR2004
            x, aux, mask = entry
            return x, mask, aux
        msg = f"Unexpected batch tuple length: {len(entry)}"
        raise ValueError(msg)

    def _coord_loss_ramp(self) -> float:
        """Linear 0→1 ramp over ``coord_loss_warmup_epochs``.

        During warmup (including epoch 0 when ``warmup > 0``) we bypass
        ``TaskWeighting`` entirely — see the class docstring note about
        ``s_c`` drifting to -∞ when coord is identically zero.
        """
        warmup = self.config.coord_loss_warmup_epochs
        if warmup <= 0:
            return 1.0
        return min(1.0, (self.current_epoch + 1) / warmup)

    def _combine_losses(
        self,
        task_weighting: TaskWeighting,
        out: dict[str, Tensor],
        *,
        coord_loss_enabled: bool,
        circle_loss_weight: float,
    ) -> Tensor:
        """Combine recon (+ coord via TaskWeighting) + commit + λ·circle.

        The ``TaskWeighting`` path is only taken once the coord ramp reaches
        1.0: during warmup we add ``ramp·coord`` directly so the learnable
        ``log_var_coord`` does not see an (initially) near-zero task.
        """
        recon = out["reconstruction_loss"]
        commit = out["commitment_loss"]
        circle = out.get("circle_loss", recon.new_zeros(()))

        if coord_loss_enabled:
            ramp = self._coord_loss_ramp()
            if ramp >= 1.0:
                base = task_weighting(recon, out["coord_loss"])
            else:
                # Plain additive during warmup (avoids TaskWeighting's
                # ``log_var_coord`` drifting to -inf under near-zero coord).
                # ``log_var_{recon,coord}`` are not referenced on this path,
                # so DDP would flag them unused — touch them with a zero
                # coefficient to keep them in the autograd graph.
                unused_ref = 0.0 * (
                    task_weighting.log_var_recon + task_weighting.log_var_coord
                )
                base = recon + ramp * out["coord_loss"] + unused_ref
        else:
            # coord_loss disabled — TaskWeighting entirely unused. Same DDP
            # workaround: reference its parameters with zero weight.
            unused_ref = 0.0 * (
                task_weighting.log_var_recon + task_weighting.log_var_coord
            )
            base = recon + unused_ref

        return base + commit + circle_loss_weight * circle

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

        # Protein VQ-VAE (sequence batch with mask, optional aux)
        if "protein" in batch:
            prot_x, prot_mask, prot_aux = self._unpack(batch["protein"])
            prot_out = self.protein_vqvae(prot_x, mask=prot_mask, aux=prot_aux)
            prot_loss = self._combine_losses(
                self.protein_task_weighting,
                prot_out,
                coord_loss_enabled=self.config.protein.coord_loss_enabled,
                circle_loss_weight=self.config.protein.circle_loss_weight,
            )
            total_loss = total_loss + prot_loss
            self.log("train/protein_recon", prot_out["reconstruction_loss"])
            self.log("train/protein_commit", prot_out["commitment_loss"])
            self.log("train/protein_circle", prot_out["circle_loss"])
            if self.config.protein.coord_loss_enabled:
                self._log_task_weighting("train/protein", self.protein_task_weighting)
                self.log("train/protein_coord", prot_out["coord_loss"])
            real_indices = prot_out["indices"][prot_mask]
            self._log_utilization("train/protein", real_indices)
            self._log_diagnostics("train/protein", prot_out["diagnostics"])

        # Ligand VQ-VAE (sequence batch with mask, optional aux)
        if "ligand" in batch:
            lig_x, lig_mask, lig_aux = self._unpack(batch["ligand"])
            lig_out = self.ligand_vqvae(lig_x, mask=lig_mask, aux=lig_aux)
            lig_loss = self._combine_losses(
                self.ligand_task_weighting,
                lig_out,
                coord_loss_enabled=self.config.ligand.coord_loss_enabled,
                circle_loss_weight=self.config.ligand.circle_loss_weight,
            )
            total_loss = total_loss + lig_loss
            self.log("train/ligand_recon", lig_out["reconstruction_loss"])
            self.log("train/ligand_commit", lig_out["commitment_loss"])
            self.log("train/ligand_circle", lig_out["circle_loss"])
            if self.config.ligand.coord_loss_enabled:
                self._log_task_weighting("train/ligand", self.ligand_task_weighting)
                self.log("train/ligand_coord", lig_out["coord_loss"])
            real_indices = lig_out["indices"][lig_mask]
            self._log_utilization("train/ligand", real_indices)
            self._log_diagnostics("train/ligand", lig_out["diagnostics"])

        self.log("train/coord_loss_ramp", self._coord_loss_ramp())

        self.log("train/total_loss", total_loss, prog_bar=True)

        opt.zero_grad()
        self.manual_backward(total_loss)
        self._log_submodule_grad_norms()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.log("train/grad_norm", grad_norm)
        opt.step()
        self._log_submodule_param_norms()
        self._log_latent_norm_gain()
        self._log_adam_v_mean(opt)
        if sch is not None:
            sch.step()
            self.log("train/lr", opt.param_groups[0]["lr"])

    def validation_step(
        self,
        batch: dict[str, tuple],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        if "protein" in batch:
            prot_x, prot_mask, prot_aux = self._unpack(batch["protein"])
            prot_out = self.protein_vqvae(prot_x, mask=prot_mask, aux=prot_aux)
            self.log(
                "val/protein_recon",
                prot_out["reconstruction_loss"],
                sync_dist=True,
            )
            self.log("val/protein_commit", prot_out["commitment_loss"], sync_dist=True)
            self.log("val/protein_circle", prot_out["circle_loss"], sync_dist=True)
            if self.config.protein.coord_loss_enabled:
                self.log("val/protein_coord", prot_out["coord_loss"], sync_dist=True)
            real_indices = prot_out["indices"][prot_mask]
            self._log_utilization("val/protein", real_indices)
            self._log_diagnostics("val/protein", prot_out["diagnostics"])

        if "ligand" in batch:
            lig_x, lig_mask, lig_aux = self._unpack(batch["ligand"])
            lig_out = self.ligand_vqvae(lig_x, mask=lig_mask, aux=lig_aux)
            self.log("val/ligand_recon", lig_out["reconstruction_loss"], sync_dist=True)
            self.log("val/ligand_commit", lig_out["commitment_loss"], sync_dist=True)
            self.log("val/ligand_circle", lig_out["circle_loss"], sync_dist=True)
            if self.config.ligand.coord_loss_enabled:
                self.log("val/ligand_coord", lig_out["coord_loss"], sync_dist=True)
            real_indices = lig_out["indices"][lig_mask]
            self._log_utilization("val/ligand", real_indices)
            self._log_diagnostics("val/ligand", lig_out["diagnostics"])

    def test_step(
        self,
        batch: dict[str, tuple],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        if "protein" in batch:
            prot_x, prot_mask, prot_aux = self._unpack(batch["protein"])
            prot_out = self.protein_vqvae(prot_x, mask=prot_mask, aux=prot_aux)
            self.log(
                "test/protein_recon",
                prot_out["reconstruction_loss"],
                sync_dist=True,
            )
            self.log("test/protein_commit", prot_out["commitment_loss"], sync_dist=True)
            if self.config.protein.coord_loss_enabled:
                self.log("test/protein_coord", prot_out["coord_loss"], sync_dist=True)
            real_indices = prot_out["indices"][prot_mask]
            self._log_utilization("test/protein", real_indices)

        if "ligand" in batch:
            lig_x, lig_mask, lig_aux = self._unpack(batch["ligand"])
            lig_out = self.ligand_vqvae(lig_x, mask=lig_mask, aux=lig_aux)
            self.log(
                "test/ligand_recon",
                lig_out["reconstruction_loss"],
                sync_dist=True,
            )
            self.log(
                "test/ligand_commit",
                lig_out["commitment_loss"],
                sync_dist=True,
            )
            if self.config.ligand.coord_loss_enabled:
                self.log("test/ligand_coord", lig_out["coord_loss"], sync_dist=True)
            real_indices = lig_out["indices"][lig_mask]
            self._log_utilization("test/ligand", real_indices)

    def _log_task_weighting(self, prefix: str, weighting: TaskWeighting) -> None:
        """Log uncertainty-weighting scalars: log σ² and effective task weights."""
        self.log(f"{prefix}_log_sigma_recon", weighting.log_var_recon.detach())
        self.log(f"{prefix}_log_sigma_coord", weighting.log_var_coord.detach())
        self.log(f"{prefix}_recon_weight", torch.exp(-weighting.log_var_recon.detach()))
        self.log(f"{prefix}_coord_weight", torch.exp(-weighting.log_var_coord.detach()))

    def _log_utilization(self, prefix: str, indices: Tensor) -> None:
        """Log codebook utilization and perplexity."""
        unique_codes = indices.unique().numel()
        total_codes = (
            self.config.protein.codebook_size
            if "protein" in prefix
            else self.config.ligand.codebook_size
        )
        utilization = unique_codes / total_codes
        self.log(f"{prefix}_codebook_util", utilization)

        # Perplexity: exp(entropy of code usage distribution)
        counts = torch.bincount(indices, minlength=total_codes).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * probs.log()).sum()
        self.log(f"{prefix}_perplexity", entropy.exp())

    def _log_diagnostics(self, prefix: str, diagnostics: dict[str, Tensor]) -> None:
        """Log per-codebook diagnostic signals for late-training collapse triage."""
        for key, value in diagnostics.items():
            self.log(f"{prefix}_{key}", value.float())

    def _submodules_for_stats(
        self,
    ) -> list[tuple[str, str, torch.nn.Module]]:
        """Modules whose weight / gradient norms we track for triage."""
        return [
            ("protein", "encoder", self.protein_vqvae.transformer_encoder),
            ("protein", "decoder", self.protein_vqvae.transformer_decoder),
            ("protein", "latent_proj", self.protein_vqvae.latent_proj),
            ("protein", "latent_norm", self.protein_vqvae.latent_norm),
            ("ligand", "encoder", self.ligand_vqvae.transformer_encoder),
            ("ligand", "decoder", self.ligand_vqvae.transformer_decoder),
            ("ligand", "latent_proj", self.ligand_vqvae.latent_proj),
            ("ligand", "latent_norm", self.ligand_vqvae.latent_norm),
        ]

    def _log_submodule_grad_norms(self) -> None:
        """Log L2 gradient norm per submodule (pre-clip, to localise explosions)."""
        for model, sub, module in self._submodules_for_stats():
            acc = torch.zeros((), device=self.device)
            for p in module.parameters():
                if p.grad is not None:
                    acc = acc + p.grad.detach().float().pow(2).sum()
            self.log(f"train/{model}_grad_norm_{sub}", acc.sqrt())

    def _log_submodule_param_norms(self) -> None:
        """Log L2 parameter norm per submodule (catches weight-norm drift)."""
        for model, sub, module in self._submodules_for_stats():
            acc = torch.zeros((), device=self.device)
            for p in module.parameters():
                acc = acc + p.detach().float().pow(2).sum()
            self.log(f"train/{model}_param_norm_{sub}", acc.sqrt())

    def _log_latent_norm_gain(self) -> None:
        """Log stats of the learnable gain (gamma) on the latent LayerNorm."""
        for name, vqvae in (
            ("protein", self.protein_vqvae),
            ("ligand", self.ligand_vqvae),
        ):
            gamma = vqvae.latent_norm.weight.detach().float()
            self.log(f"train/{name}_latent_norm_gamma_mean", gamma.mean())
            self.log(f"train/{name}_latent_norm_gamma_max", gamma.abs().max())

    def _log_adam_v_mean(self, opt: torch.optim.Optimizer) -> None:
        """Log mean of AdamW's second moment (v); spots under-estimation."""
        v_vals: list[Tensor] = []
        for group in opt.param_groups:
            for p in group["params"]:
                state = opt.state.get(p)
                if state is not None and "exp_avg_sq" in state:
                    v_vals.append(state["exp_avg_sq"].detach().float().mean())
        if v_vals:
            self.log("train/adam_v_mean", torch.stack(v_vals).mean())

    def configure_optimizers(self) -> dict:
        opt = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)

        total_steps = int(self.trainer.estimated_stepping_batches)
        # 5% of training as linear warmup, capped at 500 steps.
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

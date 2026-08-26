"""Lightning module for joint protein + ligand VQ-VAE training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lightning as L
import torch
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from prolit.tokenizers.vqvae import TransformerVQVAE

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

    from prolit.config import (
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
        self.config: AtomVQVAETrainingConfig = config
        self.save_hyperparameters()
        self.vqvae = TransformerVQVAE(config.atom)
        self.automatic_optimization: bool = False
        # Always a ParameterDict, empty unless uncertainty weighting is on. An
        # empty one contributes no state_dict keys, so a checkpoint from either
        # setting still loads strictly into a module built for the other.
        self.log_var = torch.nn.ParameterDict(
            self._build_log_vars() if self._uncertainty else {}
        )
        # Running mean of each head's raw loss, for ``loss_balancing="scale"``.
        # A buffer rather than a parameter: it tracks the loss, it is not
        # optimised, and it has to survive a resume.
        if self._balancing == "scale":
            for name in self._loss_names():
                self.register_buffer(f"_scale_{name}", torch.ones(()))
        # Lagrange multipliers, one per constrained head. Buffers rather than
        # parameters: dual ascent RAISES them when a constraint is violated,
        # which is the opposite of what gradient descent on the loss would do.
        if self._balancing == "constrained":
            for name in self._constraints():
                self.register_buffer(f"_lam_{name}", torch.ones(()))

    @property
    def _balancing(self) -> str:
        return str(getattr(self.config.atom, "loss_balancing", "none"))

    @property
    def _uncertainty(self) -> bool:
        return self._balancing == "uncertainty"

    def _loss_names(self) -> list[str]:
        """Every term ``_compute_recon_loss`` can emit, known before the first
        batch: these become parameters or buffers, which cannot appear later."""
        names = [name for name, _kind, _dim in self.vqvae.recon_heads]
        unified = getattr(self.config.atom, "pair_distance_loss", False)
        if not getattr(self.config.atom, "drop_clash", False) and (
            not unified or getattr(self.config.atom, "keep_clash", False)
        ):
            names.append("clash")
        if unified:
            names.append("pair")
        if getattr(self.config.atom, "local_distance_loss", False):
            names.append("local")
        if not unified:
            if getattr(self.config.atom, "bond_distance_loss", False):
                names += ["bond12", "bond13"]
            if getattr(self.config.atom, "distance_map_loss", False):
                names.append("dmap")
        return names

    def _constraints(self) -> dict[str, float]:
        """Heads held to the control run's level, and that level."""
        targets = dict(getattr(self.config.atom, "constraint_targets", {}) or {})
        return {k: v for k, v in targets.items() if k in set(self._loss_names())}

    def _build_log_vars(self) -> dict[str, torch.nn.Parameter]:
        """One learnable log-variance per loss term."""
        return {n: torch.nn.Parameter(torch.zeros(())) for n in self._loss_names()}

    #: Loss terms that are squared errors rather than cross-entropies. Kendall's
    #: derivation puts a factor 1/2 on the Gaussian likelihood and none on the
    #: softmax one, which is the only place the two kinds differ.
    _REGRESSION = frozenset(
        {"coord", "knn_offsets", "clash", "bond12", "bond13", "dmap", "pair", "local"}
    )

    #: Everything between the descriptor and the code. Frozen together or not
    #: at all: freezing the encoder while the codebook still moves would change
    #: the codes anyway, which is the one thing this must not do.
    _ENCODER_PARTS = (
        "cat_embeddings", "input_norm", "input_proj", "transformer_encoder",
        "latent_proj", "latent_norm", "codebook",
    )

    def _freeze_encoder(self) -> None:
        """Hold the encoder and codebook where the checkpoint left them.

        The aromatic head is a *decoder* head, and it collapsed (recall 0.014 on
        the reference ligands). Retraining the whole tokenizer to fix it would
        move the codes, and every token stream and language model downstream
        would have to be rebuilt. Retraining only the decoder cannot: the same
        descriptor still maps to the same code, so ``data/lm_tokens_*`` and the
        CLM trained on them stay valid, and only what a code decodes *to*
        improves.

        The codebook needs more than ``requires_grad = False`` -- it is updated
        by EMA inside its own forward, gated on ``self.training`` -- so it is
        held in eval mode, and :meth:`train` re-applies that after Lightning
        puts the module back in train mode each epoch.
        """
        for name in self._ENCODER_PARTS:
            part = getattr(self.vqvae, name, None)
            if part is None:
                continue
            for param in part.parameters():
                param.requires_grad_(False)  # noqa: FBT003
            part.eval()

    def train(self, mode: bool = True) -> AtomVQVAEModule:  # noqa: FBT001, FBT002
        super().train(mode)
        if getattr(self.config.atom, "freeze_encoder", False):
            for name in self._ENCODER_PARTS:
                part = getattr(self.vqvae, name, None)
                if part is not None:
                    part.eval()
        return self

    def on_fit_start(self) -> None:
        self._inject_normalization_stats()
        if getattr(self.config.atom, "freeze_encoder", False):
            self._freeze_encoder()

    def on_validation_start(self) -> None:
        self._inject_normalization_stats()

    def on_test_start(self) -> None:
        self._inject_normalization_stats()

    def _inject_normalization_stats(self) -> None:
        dm = getattr(self.trainer, "datamodule", None)
        stats = getattr(dm, "norm_stats", None) if dm is not None else None
        if stats is not None and "atom_mean" in stats and "atom_std" in stats:
            self.vqvae.set_normalization(stats["atom_mean"], stats["atom_std"])

    def _combine_losses(
        self,
        out: dict[str, Any],
        recon_weights: dict[str, float],
    ) -> Tensor:
        head_losses: dict[str, Tensor] = out["head_losses"]
        commit = out["commitment_loss"]
        if self._balancing == "scale":
            return self._scale_balanced(head_losses, commit)
        if self._balancing == "constrained":
            return self._constrained(head_losses, commit)
        if not self.log_var:
            weighted = sum(
                recon_weights.get(name, 1.0) * loss
                for name, loss in head_losses.items()
            )
            return weighted + commit
        # exp(-s) L + s/2, halved again for the regression terms. The +s/2 is
        # what stops the model from driving every weight to zero.
        total = commit
        for name, loss in head_losses.items():
            s = self.log_var.get(name)
            if s is None:
                total = total + recon_weights.get(name, 1.0) * loss
                continue
            scale = 0.5 if name in self._REGRESSION else 1.0
            total = total + scale * torch.exp(-s) * loss + 0.5 * s
        return total

    def _scale_balanced(
        self, head_losses: dict[str, Tensor], commit: Tensor
    ) -> Tensor:
        """Each head divided by a running mean of itself.

        The heads' raw magnitudes span four orders (``coord`` ~0.27 against
        ``bb_sc`` ~0.0001), so an unweighted sum is decided entirely by scale.
        Dividing by a detached running mean makes every head contribute about
        1.0, i.e. they compete on RELATIVE progress.

        MEASURED TO DIVERGE, and the reasoning that said it could not was
        wrong. What is pinned near 1 is the CONTRIBUTION; the weight is
        ``1/mean``, which grows without bound as a head's loss approaches zero,
        exactly as ``exp(-s)`` does under uncertainty weighting. Run
        ``vq_scale`` reached ``weight/aromatic`` = 7.6e6 while ``weight/coord``
        fell to 0.012, solving the chemistry to 0.99999 accuracy and abandoning
        the geometry (val/atom_coord 85.8 against the control's 0.0958). Kept so
        the failure is on the record, not for use.
        """
        decay = float(getattr(self.config.atom, "loss_scale_decay", 0.99))
        total = commit
        for name, loss in head_losses.items():
            key = f"_scale_{name}"
            if not hasattr(self, key):
                total = total + loss
                continue
            scale: Tensor = getattr(self, key)
            if self.training:
                scale.mul_(decay).add_(loss.detach().clamp_min(1e-8), alpha=1 - decay)
            total = total + loss / scale.clamp_min(1e-8)
        return total

    def _constrained(
        self, head_losses: dict[str, Tensor], commit: Tensor
    ) -> Tensor:
        """Geometry as the objective, chemistry as constraints at the control's level.

        Both balancers aim at equal contribution, which is not the goal: the goal
        is to improve geometry WITHOUT regressing chemistry, an asymmetric
        requirement no balance can express. Here the geometry heads keep their
        weights and each chemistry head carries a multiplier that dual ascent
        raises while its loss is above the level ``vq_ctrl_p3`` reached, and lets
        fall once it is not.

        Two bounds keep this from ending like uncertainty weighting, which
        diverged because nothing capped a weight:

        * the multiplier lives in ``[0, 1]``. A satisfied constraint releases its
          head entirely, so the model may spend that capacity on geometry; a
          violated one is held at weight 1 and no higher. The worst case is
          therefore the unweighted sum, and since every hand weight this replaces
          was <= 0.5, weight 1 is strictly more pressure than the control run
          needed to reach these very levels -- the constraints are reachable
          inside the bound, not merely hoped to be.
        * the violation is clipped to +-1 before it is applied, so a multiplier
          moves by at most ``constraint_lr`` per step. Without it the ratio at
          initialisation (cross-entropy ~2.3 against a target of 0.002) would
          saturate every multiplier on the first batch and the scheme would
          degenerate to on/off.
        """
        targets = self._constraints()
        lr = float(getattr(self.config.atom, "constraint_lr", 0.01))
        total = commit
        for name, loss in head_losses.items():
            target = targets.get(name)
            if target is None:  # objective: geometry, at its configured weight
                total = total + self.config.atom.recon_weights.get(name, 1.0) * loss
                continue
            lam: Tensor = getattr(self, f"_lam_{name}")
            if self.training:
                violation = (loss.detach() - target) / max(target, 1e-6)
                lam.add_(lr * violation.clamp(-1.0, 1.0)).clamp_(0.0, 1.0)
            total = total + lam * loss
        return total

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
        if train and self._balancing == "constrained":
            for name in self._constraints():
                self.log(f"weight/{name}", getattr(self, f"_lam_{name}"))
        if train and (self.log_var or self._balancing == "scale"):
            # The balance that replaces the hand-set weights is the interesting
            # artefact, so it has to be inspectable after the fact.
            for name, s in self.log_var.items():
                self.log(f"weight/{name}", torch.exp(-s.detach()))
            for name in self._loss_names():
                key = f"_scale_{name}"
                if hasattr(self, key):
                    self.log(f"weight/{name}", 1.0 / getattr(self, key).clamp_min(1e-8))
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
        if isinstance(sch, list):  # pragma: no cover - one scheduler is configured
            msg = "Expected a single LR scheduler"
            raise TypeError(msg)

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

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
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

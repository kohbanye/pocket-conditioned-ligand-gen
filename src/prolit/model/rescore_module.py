"""LightningModule for the pose-scoring head (discriminative rescorer).

The pretrained complex-token MLM encoder (:class:`~prolit.model.mlm.ProLITMLM`,
warm-started from an MLM checkpoint) is fine-tuned with a small MLP head that
mean-pools the ligand-token representations and regresses the pose RMSD. Lower
predicted RMSD = more native-like -- so at eval time a pose is scored by
``-rmsd_pred``. This is the discriminative complement to the zero-shot masked
PLL (:mod:`prolit.model.mlm_score`): it optimises pose ranking directly.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import lightning as L
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    SequentialLR,
)

from prolit.model.mlm import build_prolit_mlm, count_parameters
from prolit.tokenizers.lm_vocab import NUM_SPECIAL

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

    from prolit.config import RescoreTrainingConfig

logger = logging.getLogger(__name__)

#: Which encoder states the head averages. All three are parameter-free, so one
#: checkpoint's weights can be read under any of them -- which is the point:
#: they separate "the representation does not hold it" from "the readout does
#: not reach it". ``ligand`` is what every trained checkpoint used and stays the
#: default; a checkpoint trained under another mode must be loaded under that
#: same mode, or its weights are read through pooling they never saw.
#: ``concat`` is the one readout that is NOT parameter-free: it keeps the pocket
#: mean and the ligand mean side by side, so the head's first layer reads 2h.
#: A checkpoint loaded under the wrong readout therefore fails with a shape
#: error instead of silently scoring weights through pooling they never saw.
READOUTS = ("ligand", "balanced", "allatoms", "concat")


def batch_correlation_loss(pred: Tensor, target: Tensor) -> Tensor:
    """``1 - Pearson(pred, target)`` across the batch, in [0, 2].

    Scale-free by construction: an affine rescaling of the prediction leaves it
    unchanged, so unlike squared error it cannot be reduced by shrinking toward
    the mean. That is the point -- the CASF diagnosis found target-level spread
    compressed by ~19%, which a regression objective rewards whenever the signal
    is uncertain.

    A batch whose predictions or labels are constant has no correlation to
    speak of; the term is zero there rather than NaN, so one degenerate batch
    cannot poison the epoch.
    """
    p = pred - pred.mean()
    t = target - target.mean()
    denom = p.norm() * t.norm()
    if float(denom) <= 1e-8:  # noqa: PLR2004
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    return 1.0 - (p * t).sum() / denom


def group_mean_loss(pred: Tensor, target: Tensor, group_ids: Tensor) -> Tensor:
    """Smooth-L1 between each group's MEAN prediction and its mean label.

    A group is one protein. The per-complex regression term already penalises
    every residual, but it cannot tell a group that is uniformly offset from one
    whose errors scatter around the right level -- and on CASF the offset is
    what costs us: target-level differences come out ~19% too small. Averaging
    inside the group first makes the systematic part of the error its own term.

    Groups of one contribute their own residual, which the regression term
    already covers; they are kept rather than filtered so the value does not
    depend on how a batch happens to be composed.
    """
    ids = torch.unique(group_ids)
    pm = torch.stack([pred[group_ids == g].mean() for g in ids])
    tm = torch.stack([target[group_ids == g].mean() for g in ids])
    return nn.functional.smooth_l1_loss(pm, tm)


class ComplexRescoreModule(L.LightningModule):
    """MLM encoder + mean-pool over ligand tokens + MLP -> predicted RMSD."""

    def __init__(
        self,
        config: RescoreTrainingConfig,
        mlm_state: dict | None = None,
        readout: str = "ligand",
        layer_mix: bool = False,  # noqa: FBT001, FBT002
        lora_rank: int = 0,
    ) -> None:
        super().__init__()
        self.config: RescoreTrainingConfig = config
        if not 0 < getattr(config, "encoder_lr_scale", 1.0) <= 1:
            msg = "encoder_lr_scale must be positive and at most 1"
            raise ValueError(msg)
        if readout not in READOUTS:
            msg = f"unknown readout {readout!r}; expected one of {READOUTS}"
            raise ValueError(msg)
        # An init argument rather than a config field: the config INSTANCE is
        # pickled into every existing checkpoint, so adding a field there would
        # make old checkpoints load without it and fail on first access.
        # ty cannot type assignment through LightningModule's __setattr__; these
        # are plain Python attributes, not parameters, so no state_dict key moves.
        self.readout = readout  # ty: ignore[unresolved-attribute]
        # Same reasoning as `readout`: an init argument, not a config field, so
        # existing checkpoints (whose config INSTANCE is pickled) still load.
        self.layer_mix = bool(layer_mix)  # ty: ignore[unresolved-attribute]
        self.lora_rank = int(lora_rank)  # ty: ignore[unresolved-attribute]
        if self.lora_rank and getattr(config, "encoder_lr_scale", 1.0) != 1.0:
            msg = "lora_rank with encoder_lr_scale is untested"
            raise ValueError(msg)
        if self.layer_mix and getattr(config, "encoder_lr_scale", 1.0) != 1.0:
            msg = (
                "layer_mix with encoder_lr_scale is untested: the scaled optimizer "
                "path builds its own groups and would not give the mixing weights "
                "their own learning rate"
            )
            raise ValueError(msg)
        self.save_hyperparameters(ignore=["mlm_state"])
        self.encoder = build_prolit_mlm(config.model)
        if mlm_state is not None:
            # MLM checkpoint keys are prefixed "model." (ProLITMLMModule.model).
            enc_state = {
                k[len("model.") :]: v
                for k, v in mlm_state.items()
                if k.startswith("model.")
            }
            miss, unexp = self.encoder.load_state_dict(enc_state, strict=False)
            logger.info("warm-start: %d missing, %d unexpected", len(miss), len(unexp))
        if self.layer_mix:
            # ELMo-style scalar mix over the embedding output and all layers.
            # Created ONLY when asked for, so a checkpoint trained without it
            # keeps its exact state_dict and one trained with it fails loudly
            # rather than silently reading weights it never saw. Zeros mean the
            # mix starts as the uniform average of every layer; the encoder
            # stays frozen, so these are the only new trainable parameters.
            init = torch.zeros(config.model.num_hidden_layers + 1)
            # Zeros are the uniform average, which the first round measured as
            # worse than the last layer alone
            # (docs/results/2026-09-15_layer_mix_uniform_refuted.md). A positive
            # last entry starts the mix where a last-layer head already is.
            init[-1] = float(getattr(config, "layer_mix_init_last", 0.0))
            self.layer_weights = nn.Parameter(init)
        h = config.model.hidden_size
        # "concat" hands the head [pocket mean, ligand mean]; every other readout
        # reduces the states to a single h vector.
        in_dim = 2 * h if readout == "concat" else h
        self.head = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Dropout(config.head_dropout),
            nn.LayerNorm(h),
            nn.Linear(h, 1),
        )
        self._logged: bool = False
        self._attach_lora_and_freeze(config)

    def _attach_lora_and_freeze(self, config: RescoreTrainingConfig) -> None:
        """Attach LoRA adapters, then freeze whatever should not train.

        Order matters. The adapters go on AFTER the warm start, so the MLM
        checkpoint still loads against the plain encoder's state_dict; the
        freeze runs after that, so the adapters -- which live under
        ``self.encoder`` -- have to be switched back on or the arm would train
        nothing but its head.
        """
        if self.lora_rank:
            from prolit.model.lora import apply_lora  # noqa: PLC0415

            apply_lora(
                self.encoder, self.lora_rank, float(getattr(config, "lora_alpha", 1.0))
            )
        if config.freeze_encoder:
            self._freeze_encoder()
        if self.lora_rank:
            for name, param in self.encoder.named_parameters():
                if "lora_" in name:
                    param.requires_grad_(True)  # named `mode`, not `requires_grad`

    def _freeze_encoder(self) -> None:
        """Train the head only: 0.59M parameters instead of 99.6M.

        The encoder is put in eval mode as well as frozen, because a frozen
        module still has dropout active in train mode and would hand the head a
        different vector for the same complex on every step -- noise the head
        cannot fit and is not meant to. :meth:`train` keeps it that way; see
        there for why that override is required rather than tidy.
        """
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> ComplexRescoreModule:  # noqa: FBT001, FBT002
        """Lightning calls ``train()`` on the whole module at each epoch.

        ``nn.Module.train`` recurses into every child, so without this the
        encoder is put back into train mode at the start of every epoch and
        starts applying dropout again -- frozen weights, but a moving output.
        The parameters would still not update, so nothing would look wrong.
        """
        super().train(mode)
        if self.config.freeze_encoder:
            self.encoder.eval()
        return self

    def _pool(self, hs: Tensor, batch: dict[str, Tensor]) -> Tensor:
        """Mean of the encoder states over the ligand tokens -> one (B, H) vector.

        Only the ligand side is read: every pose of a target carries the SAME
        pocket token ids (``p_codes`` is built once per target and reused), so
        the pocket rows hold nothing that separates one pose from another. The
        ligand rows have already attended to the pocket, so what is averaged is
        "the ligand atom, having seen its environment".

        Dividing by the ligand-token count rather than the sequence length makes
        the vector a per-atom average, so a 50-atom ligand and a 15-atom one
        arrive on the same scale and the head needs no size-dependent calibration.

        Mean is the whole of it. Concatenating a max pool (the worst atom, which
        a mean over ~30 atoms dilutes) was measured and is worse -- 89.5/56.8
        against 92.6/74.7 DP@2A/DP@1A on a 95-target probe -- because taking one
        atom per channel discards the rest. Learned attention pooling, ligand->
        pocket cross-attention and a GenScore-style sum over ligand-pocket pairs
        were each built and measured; none beat this.
        """
        if self.readout == "ligand":
            m = batch["ligand_mask"].unsqueeze(-1).to(hs.dtype)  # (B,L,1)
            return (hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        protein, ligand = self._component_masks(batch)
        pm = protein.unsqueeze(-1).to(hs.dtype)
        lm = ligand.unsqueeze(-1).to(hs.dtype)
        p_sum, l_sum = (hs * pm).sum(dim=1), (hs * lm).sum(dim=1)
        p_n, l_n = pm.sum(dim=1), lm.sum(dim=1)
        if self.readout == "balanced":
            # Equal weight per SIDE, so a 200-atom pocket does not outvote a
            # 25-atom ligand simply by being larger.
            return (p_sum / p_n + l_sum / l_n) / 2
        if self.readout == "concat":
            # Pocket and ligand kept separate. The "ligand" default was argued for
            # POSE rescoring, where every pose of a target shares the pocket rows,
            # so the pocket cannot separate poses. For AFFINITY across targets the
            # pocket is exactly what separates one target's typical affinity from
            # another's: on CASF-2016 the ligand readout loses the between-target
            # part of R and pooling all atoms recovers it but dilutes the ligand
            # signal that orders ligands within a target
            # (docs/notes/2026-09-14_prereg_concat_readout.md). Concatenating
            # keeps both without either averaging the other away.
            return torch.cat([p_sum / p_n, l_sum / l_n], dim=-1)
        return (p_sum + l_sum) / (p_n + l_n)  # "allatoms": equal weight per ATOM

    def _component_masks(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Protein-atom and ligand-atom masks, from the token ids themselves.

        The protein side is derived rather than passed: it is every real atom
        token the dataset did not mark as ligand. Markers, padding and the
        specials are excluded by the code range, so nothing but atoms is
        averaged. A complex with no protein or no ligand atom would make one of
        the two means a division by zero, which is a corrupt batch rather than
        an edge case worth silently clamping.
        """
        tokens, ligand = batch["input_ids"], batch["ligand_mask"].bool()
        attention = batch["attention_mask"].bool()
        codebook = self.config.model.atom_codebook_size
        atoms = (
            (tokens >= NUM_SPECIAL) & (tokens < NUM_SPECIAL + codebook) & attention
        )
        protein = atoms & ~ligand
        if not bool((protein.any(dim=1) & ligand.any(dim=1)).all()):
            msg = "a complex has no protein or no ligand atom token"
            raise ValueError(msg)
        return protein, ligand

    def _encode_states(self, batch: dict[str, Tensor]) -> Tensor:
        """Encoder states the readout pools, (B, L, H).

        Every head before this read the last layer only, which is the layer the
        MLM objective specialised. With ``layer_mix`` the head instead learns a
        softmax over the embedding output and all layers.
        """
        if not self.layer_mix:
            return self.encoder.encode(
                batch["input_ids"], batch["attention_mask"]
            )  # (B,L,H)
        states = self.encoder.encode(
            batch["input_ids"], batch["attention_mask"], return_all_layers=True
        )  # (1+layers, B, L, H)
        weights = torch.softmax(self.layer_weights, dim=0).to(states.dtype)
        return (states * weights[:, None, None, None]).sum(dim=0)

    def _predict(self, pooled: Tensor) -> Tensor:
        """Predicted RMSD in angstroms, one scalar per pose. Lower is better."""
        return self.head(pooled).squeeze(-1)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        hs = self._encode_states(batch)
        return self._predict(self._pool(hs, batch))  # (B,) predicted label

    def on_fit_start(self) -> None:
        if not self._logged:
            self.print(f"Rescorer parameters: {count_parameters(self) / 1e6:.1f}M")
            self._logged: bool = True

    def _listwise_loss(self, pred: Tensor, rmsd: Tensor, groups: Tensor) -> Tensor:
        """ListNet cross-entropy within each group: match the softmax over the
        predicted scores to a softmax over the labels.

        ``listwise_higher_is_better`` picks the sign. A pose group is labelled
        by RMSD, where small is good, so the default negates both sides and the
        softmax concentrates on the near-native end. An affinity group is
        labelled by pK, where large is good; without the flip the same
        expression would spend the term on telling the weakest binders apart.

        Docking power asks "which pose in this set is the native one", which a
        per-pose regression only optimizes indirectly -- and a pairwise margin
        loss optimizes too literally (every pair counts the same, including the
        6 A vs 8 A pairs nobody cares about). A soft label temperature of a few
        tenths of an angstrom puts the whole loss on the near-native end while
        the model stays a per-pose scorer at inference.

        A top-k variant of this term was tried and dropped (PR #14). Restricting
        the softmax to the k poses the head ranks highest measured 90.5 -> 89.5
        on CASF: it won 10 of the 24 targets whose answer sat at rank 2-5, which
        is what it was for, but lost 13 that the full-set term had been ordering
        correctly -- with no gradient left for the far poses, the global ranking
        decays. Selecting the k by *label* instead was worse still (87.7 when
        added to the full-set term), the more weight it carried the worse it
        got, which is the signature of a self-referential target rather than of
        a gradient budget.
        """
        tau_l = self.config.listwise_label_tau
        sign = 1.0 if self.config.listwise_higher_is_better else -1.0
        total = pred.new_zeros(())
        n = 0
        for g in groups.unique():
            m = groups == g
            p, r = pred[m], rmsd[m]
            if p.numel() < 2:  # noqa: PLR2004
                continue
            target = torch.softmax(sign * r.float() / tau_l, dim=0)
            logp = torch.log_softmax(
                sign * p.float() / self.config.listwise_pred_tau, dim=0
            )
            total = total - (target * logp).sum()
            n += 1
        return total / max(1, n)

    def _step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        pred = self(batch)
        rmsd = batch["rmsd"]
        target = rmsd.clamp(max=self.config.rmsd_cap)
        reg = nn.functional.smooth_l1_loss(pred, target)
        loss = reg
        sync = stage != "train"
        corr_w = getattr(self.config, "corr_weight", 0.0)
        if corr_w > 0:
            # OnionNet-2's shape: the two terms trade off rather than stack, so
            # the regression term is downweighted by the same amount.
            corr = batch_correlation_loss(pred, target)
            loss = (1.0 - corr_w) * reg + corr_w * corr
            self.log(f"{stage}/corr", corr, prog_bar=True, sync_dist=sync)
        distill_w = float(getattr(self.config, "distill_weight", 0.0))
        if distill_w > 0 and "teacher" in batch:
            # Reweights whatever objective is in `loss` so far (regression, or
            # the corr-weighted blend), rather than stacking another term on it.
            kd = nn.functional.smooth_l1_loss(pred, batch["teacher"])
            loss = (1.0 - distill_w) * loss + distill_w * kd
            self.log(f"{stage}/kd", kd, prog_bar=True, sync_dist=sync)
        if getattr(self.config, "group_mean_weight", 0.0) > 0 and "group_ids" in batch:
            gm = group_mean_loss(pred, target, batch["group_ids"])
            loss = loss + self.config.group_mean_weight * gm
            self.log(f"{stage}/groupmean", gm, prog_bar=True, sync_dist=sync)
        if self.config.listwise_loss_weight > 0 and "group_ids" in batch:
            lw = self._listwise_loss(pred, rmsd, batch["group_ids"])
            loss = loss + self.config.listwise_loss_weight * lw
            # Logged here, not in the dropped top-k branch it used to sit in,
            # where it only reached the logs when that term was switched on.
            self.log(f"{stage}/list", lw, prog_bar=True, sync_dist=sync)
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=sync)
        self.log(
            f"{stage}/mae",
            (pred - target).abs().mean(),
            prog_bar=stage == "val",
            sync_dist=sync,
        )
        return loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:  # noqa: ARG002
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:  # noqa: ARG002
        self._step(batch, "val")

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        if getattr(self.config, "encoder_lr_scale", 1.0) != 1.0:
            return self._configure_scaled_optimizer()
        opt = torch.optim.AdamW(
            self._param_groups(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
        )
        total = int(self.trainer.estimated_stepping_batches)
        warmup = max(1, min(self.config.warmup_steps, total - 1))
        sched = SequentialLR(
            opt,
            schedulers=[
                LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup),
                CosineAnnealingLR(
                    opt,
                    T_max=max(1, total - warmup),
                    eta_min=self.config.learning_rate * self.config.min_lr_ratio,
                ),
            ],
            milestones=[warmup],
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }

    def _param_groups(self) -> list[dict]:
        """Optimizer groups; the mixing scalars get their own learning rate.

        Fourteen scalars behind a frozen encoder do not move at the head's rate:
        the first layer-mix round finished with them still uniform to three
        decimals (docs/results/2026-09-15_layer_mix_uniform_refuted.md), so the
        round measured an unweighted average instead of a learned choice.
        """
        mix = [self.layer_weights] if self.layer_mix else []
        mix_ids = {id(p) for p in mix}
        decay, no_decay = [], []
        for param in self.parameters():
            if not param.requires_grad or id(param) in mix_ids:
                continue
            (decay if param.ndim >= 2 else no_decay).append(param)  # noqa: PLR2004
        groups = [
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        if mix:
            scale = float(getattr(self.config, "layer_mix_lr_scale", 1.0))
            groups.append(
                {
                    "params": mix,
                    "weight_decay": 0.0,
                    "lr": self.config.learning_rate * scale,
                }
            )
        return groups

    def _configure_scaled_optimizer(self) -> OptimizerLRSchedulerConfig:
        """Use proportional encoder/head LR schedules for the opt-in arm."""
        scale = self.config.encoder_lr_scale
        if not 0 < scale <= 1:
            msg = "encoder_lr_scale must be positive and at most 1"
            raise ValueError(msg)
        encoder_ids = {id(p) for p in self.encoder.parameters()}
        grouped: dict[tuple[bool, bool], list] = {}
        for param in self.parameters():
            if param.requires_grad:
                key = (id(param) in encoder_ids, param.ndim >= 2)  # noqa: PLR2004
                grouped.setdefault(key, []).append(param)
        groups = [
            {
                "params": params,
                "lr": self.config.learning_rate * (scale if is_encoder else 1.0),
                "weight_decay": self.config.weight_decay if decay else 0.0,
            }
            for (is_encoder, decay), params in grouped.items()
        ]
        opt = torch.optim.AdamW(
            groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
        )
        total = int(self.trainer.estimated_stepping_batches)
        warmup = max(1, min(self.config.warmup_steps, total - 1))

        def factor(step: int) -> float:
            if step < warmup:
                return 1e-3 + (1.0 - 1e-3) * step / warmup
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            floor = self.config.min_lr_ratio
            return floor + (1.0 - floor) * (1.0 + math.cos(math.pi * progress)) / 2

        # A scalar eta_min would give every group the same absolute floor,
        # erasing the desired encoder/head ratio at the end of the cosine.
        sched = LambdaLR(opt, lr_lambda=factor)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }


__all__ = ["ComplexRescoreModule"]

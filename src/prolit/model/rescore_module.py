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
from typing import TYPE_CHECKING

import lightning as L
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from prolit.model.mlm import build_prolit_mlm, count_parameters

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

    from prolit.config import RescoreTrainingConfig

logger = logging.getLogger(__name__)


class ComplexRescoreModule(L.LightningModule):
    """MLM encoder + mean-pool over ligand tokens + MLP -> predicted RMSD."""

    def __init__(
        self, config: RescoreTrainingConfig, mlm_state: dict | None = None
    ) -> None:
        super().__init__()
        self.config: RescoreTrainingConfig = config
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
        h = config.model.hidden_size
        self.head = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(config.head_dropout),
            nn.LayerNorm(h),
            nn.Linear(h, 1),
        )
        self._logged: bool = False

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
        m = batch["ligand_mask"].unsqueeze(-1).to(hs.dtype)  # (B,L,1)
        return (hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def _encode_states(self, batch: dict[str, Tensor]) -> Tensor:
        return self.encoder.encode(
            batch["input_ids"], batch["attention_mask"]
        )  # (B,L,H)

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
        """ListNet cross-entropy within each complex: match the softmax over the
        predicted scores to a softmax over ``-rmsd``.

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
        total = pred.new_zeros(())
        n = 0
        for g in groups.unique():
            m = groups == g
            p, r = pred[m], rmsd[m]
            if p.numel() < 2:  # noqa: PLR2004
                continue
            target = torch.softmax(-r.float() / tau_l, dim=0)
            logp = torch.log_softmax(-p.float() / self.config.listwise_pred_tau, dim=0)
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
        decay, no_decay = [], []
        for param in self.parameters():
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


__all__ = ["ComplexRescoreModule"]

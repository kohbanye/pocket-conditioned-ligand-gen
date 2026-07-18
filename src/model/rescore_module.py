"""LightningModule for the pose-scoring head (discriminative rescorer).

The pretrained complex-token MLM encoder (:class:`~src.model.complex_mlm.ComplexMLM`,
warm-started from an MLM checkpoint) is fine-tuned with a small MLP head that
mean-pools the ligand-token representations and regresses the pose RMSD. Lower
predicted RMSD = more native-like -- so at eval time a pose is scored by
``-rmsd_pred``. This is the discriminative complement to the zero-shot masked
PLL (:mod:`src.model.mlm_score`): it optimises pose ranking directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import lightning as L
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.model.complex_mlm import build_complex_mlm, count_parameters

if TYPE_CHECKING:
    from src.config import RescoreTrainingConfig

logger = logging.getLogger(__name__)


class ComplexRescoreModule(L.LightningModule):
    """MLM encoder + mean-pool over ligand tokens + MLP -> predicted RMSD."""

    def __init__(
        self, config: RescoreTrainingConfig, mlm_state: dict | None = None
    ) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=["mlm_state"])
        self.encoder = build_complex_mlm(config.model)
        if mlm_state is not None:
            # MLM checkpoint keys are prefixed "model." (ComplexMLMModule.model).
            enc_state = {
                k[len("model.") :]: v
                for k, v in mlm_state.items()
                if k.startswith("model.")
            }
            miss, unexp = self.encoder.load_state_dict(enc_state, strict=False)
            logger.info("warm-start: %d missing, %d unexpected", len(miss), len(unexp))
        h = config.model.hidden_size
        self.pooling = config.pooling
        # meanmax and xattn both concatenate a second pooled vector with the
        # ligand mean, so the head sees 2H; mean/attn produce a single H vector.
        in_dim = 2 * h if self.pooling in ("meanmax", "xattn") else h
        if self.pooling == "attn":
            # learnable per-ligand-token weighting (softmax-attention pool)
            self.attn_score = nn.Linear(h, 1)
        if self.pooling == "xattn":
            # Explicit protein-ligand interaction: a fresh trainable cross-
            # attention layer lets each ligand token attend to the pocket tokens
            # (the encoder's own attention is spent on masked-token prediction,
            # not affinity). Its pooled output is concatenated with the ligand mean.
            self.xattn = nn.MultiheadAttention(h, num_heads=12, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Dropout(config.head_dropout),
            nn.LayerNorm(h),
            nn.Linear(h, 1),
        )
        if getattr(config, "freeze_encoder", False):
            # Freeze the 99M-param encoder so only the pooling + MLP head learn.
            # A ranking loss trained end-to-end memorized the tiny affinity corpus
            # (train/rank -> 0, val/rank 0.5); with the encoder fixed the head
            # can only re-weight existing features, so the ordering signal has a
            # chance to generalize instead of overfitting.
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False
        self._logged = False

    def _pool(self, hs: Tensor, batch: dict[str, Tensor]) -> Tensor:
        """Collapse the (B, L, H) encoder states to one (B, in_dim) vector over
        the ligand tokens. Aggregation is chosen by ``self.pooling``."""
        lig = batch["ligand_mask"]
        m = lig.unsqueeze(-1).to(hs.dtype)  # (B,L,1)
        mean = (hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        if self.pooling == "mean":
            return mean
        if self.pooling == "meanmax":
            # max over ligand tokens only; mask others to -inf so they never win.
            neg = torch.finfo(hs.dtype).min
            mx = hs.masked_fill(~lig.unsqueeze(-1), neg).max(dim=1).values
            return torch.cat([mean, mx], dim=-1)
        if self.pooling == "attn":
            # softmax in fp32: bf16 masked_fill with -inf overflows on conversion.
            scores = self.attn_score(hs).squeeze(-1).float()  # (B,L)
            scores = scores.masked_fill(~lig, float("-inf"))
            w = torch.softmax(scores, dim=1).unsqueeze(-1).to(hs.dtype)  # (B,L,1)
            return (hs * w).sum(dim=1)
        if self.pooling == "xattn":
            # ligand tokens (query) attend to pocket tokens (key/value). A row
            # always has pocket tokens, so key_padding_mask is never all-True
            # (which would NaN the softmax); ligand-only complexes don't occur.
            pocket = batch["attention_mask"].bool() & ~lig  # (B,L)
            refined, _ = self.xattn(
                hs, hs, hs, key_padding_mask=~pocket, need_weights=False
            )
            rm = (refined * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            return torch.cat([mean, rm], dim=-1)
        msg = f"unknown pooling: {self.pooling}"
        raise ValueError(msg)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        hs = self.encoder.encode(batch["input_ids"], batch["attention_mask"])  # (B,L,H)
        return self.head(self._pool(hs, batch)).squeeze(-1)  # (B,) predicted label

    def on_fit_start(self) -> None:
        if not self._logged:
            self.print(f"Rescorer parameters: {count_parameters(self) / 1e6:.1f}M")
            self._logged = True

    def _ranking_loss(self, pred: Tensor, rmsd: Tensor, groups: Tensor) -> Tensor:
        """Pairwise margin loss within each complex: for poses i,j with
        rmsd_i < rmsd_j, push pred_i below pred_j by at least ``margin``.
        Directly optimizes the pose ordering that docking power measures."""
        margin = self.config.ranking_margin
        total = pred.new_zeros(())
        npairs = 0
        for g in groups.unique():
            m = groups == g
            p, r = pred[m], rmsd[m]
            if p.numel() < 2:  # noqa: PLR2004
                continue
            dp = p.unsqueeze(0) - p.unsqueeze(1)  # pred_j - pred_i  (rows i, cols j)
            dr = r.unsqueeze(0) - r.unsqueeze(1)  # rmsd_j - rmsd_i
            # pairs where i is the better pose (rmsd_i < rmsd_j): want pred_i < pred_j
            better = dr > 0
            if not better.any():
                continue
            # penalize pred_i >= pred_j - margin  ->  relu(margin - (pred_j - pred_i))
            hinge = torch.relu(margin - dp)[better]
            total = total + hinge.sum()
            npairs += int(better.sum())
        return total / max(1, npairs)

    def _step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        pred = self(batch)
        rmsd = batch["rmsd"]
        target = rmsd.clamp(max=self.config.rmsd_cap)
        reg = nn.functional.smooth_l1_loss(pred, target)
        loss = reg
        sync = stage != "train"
        if self.config.ranking_loss_weight > 0 and "group_ids" in batch:
            rank = self._ranking_loss(pred, rmsd, batch["group_ids"])
            loss = reg + self.config.ranking_loss_weight * rank
            self.log(f"{stage}/rank", rank, prog_bar=True, sync_dist=sync)
            self.log(f"{stage}/reg", reg, sync_dist=sync)
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=sync)
        self.log(
            f"{stage}/mae",
            (pred - target).abs().mean(),
            prog_bar=stage == "val",
            sync_dist=sync,
        )
        return loss

    def train(self, mode: bool = True):  # noqa: ANN201, FBT001, FBT002
        """Keep a frozen encoder in eval mode (no dropout) even when Lightning
        flips the module to train() at each epoch, so its features stay fixed."""
        super().train(mode)
        if getattr(self.config, "freeze_encoder", False):
            self.encoder.eval()
        return self

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:  # noqa: ARG002
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:  # noqa: ARG002
        self._step(batch, "val")

    def configure_optimizers(self) -> dict:
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

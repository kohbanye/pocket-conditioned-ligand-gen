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
        if getattr(config, "freeze_encoder", False):
            # Freeze the 99M-param encoder so only the MLP head learns.
            # A ranking loss trained end-to-end memorized the tiny affinity corpus
            # (train/rank -> 0, val/rank 0.5); with the encoder fixed the head
            # can only re-weight existing features, so the ordering signal has a
            # chance to generalize instead of overfitting.
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False
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

    def _listwise_loss(
        self, pred: Tensor, rmsd: Tensor, groups: Tensor, topk: int = 0
    ) -> Tensor:
        """ListNet cross-entropy within each complex: match the softmax over the
        predicted scores to a softmax over ``-rmsd``.

        Docking power asks "which pose in this set is the native one", which a
        per-pose regression only optimizes indirectly -- and a pairwise margin
        loss optimizes too literally (every pair counts the same, including the
        6 A vs 8 A pairs nobody cares about). A soft label temperature of a few
        tenths of an angstrom puts the whole loss on the near-native end while
        the model stays a per-pose scorer at inference.
        """
        tau_l = self.config.listwise_label_tau
        total = pred.new_zeros(())
        n = 0
        for g in groups.unique():
            m = groups == g
            p, r = pred[m], rmsd[m]
            if p.numel() < 2:  # noqa: PLR2004
                continue
            k = topk
            if k and p.numel() > k:
                if self.config.listwise_topk_by_label:
                    # Pick the k poses that ARE best, not the k the head thinks
                    # are. Selecting on the head's own output is self-referential:
                    # early in training it picks the wrong five and then sharpens
                    # itself on them. Measured on CASF, model-side selection cost
                    # 90.5 -> 89.5 replacing the full-set term and 87.7 added to
                    # it -- the more weight the self-selected term carried, the
                    # worse it got, which is the signature of that feedback and
                    # not of a gradient budget. The label is only used to choose
                    # the comparison set during training; inference is unchanged.
                    sel = torch.topk(r, k, largest=False).indices
                    p, r = p[sel], r[sel]
                    # A separate, sharper temperature for this term. The five
                    # poses it compares span a median 0.30 A, and at the shared
                    # tau of 0.5 a softmax over that range is nearly flat -- the
                    # term asks the head to rank them while barely preferring
                    # one. Narrowing the set instead (k=3) raised the 0-0.5 A
                    # count 58 -> 62 but cost DP@2A 91.2 -> 89.5, because the
                    # poses dropped from the set stop being ordered at all.
                    # Sharpening keeps all five in play and widens the target
                    # gaps between them.
                    tl = self.config.listwise_topk_tau or tau_l
                    target = torch.softmax(-r.float() / tl, dim=0)
                    logp = torch.log_softmax(
                        -p.float() / self.config.listwise_pred_tau, dim=0
                    )
                    total = total - (target * logp).sum()
                    n += 1
                    continue
                # Restrict the softmax to the poses the head currently ranks
                # highest. Measured on CASF, the head's top-10 overlaps
                # RTMScore's by 6 of 10 on the targets it loses -- the same
                # overlap it has on the targets it wins. It finds the right
                # candidates and mis-orders the top of them: on 1mq6 its top
                # five are 2.1, 1.0, 0.5, 0.8, 0.7 A. A softmax over all ~80
                # poses spends most of its gradient separating the 8 A decoys
                # nobody confuses, so the contest that actually decides docking
                # power gets a vanishing share of it. Taking the top k makes
                # every term a comparison the benchmark could turn on.
                sel = torch.topk(p.detach(), k, largest=False).indices
                p, r = p[sel], r[sel]
            target = torch.softmax(-r.float() / tau_l, dim=0)
            logp = torch.log_softmax(-p.float() / self.config.listwise_pred_tau, dim=0)
            total = total - (target * logp).sum()
            n += 1
        return total / max(1, n)

    def _mlm_aux_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Masked-LM loss on the same complexes: mask a fraction of the structure
        (codebook) tokens and predict them through the encoder's MLM head. Used
        as a regularizer so a ranking loss adapts the encoder to affinity without
        collapsing the pretrained structure representation."""
        ids = batch["input_ids"]
        attn = batch["attention_mask"].bool()
        cb = self.config.model.atom_codebook_size or 0
        maskable = attn & (ids < cb)  # only real structure tokens
        rand = torch.rand(ids.shape, device=ids.device)
        do = (rand < self.config.mlm_aux_mask_prob) & maskable
        if not bool(do.any()):
            return ids.new_zeros((), dtype=torch.float)
        labels = ids.masked_fill(~do, -100)
        masked = ids.masked_fill(do, self.config.model.mask_token_id)
        return self.encoder(masked, batch["attention_mask"], labels=labels).loss

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
        if self.config.listwise_loss_weight > 0 and "group_ids" in batch:
            lw = self._listwise_loss(pred, rmsd, batch["group_ids"])
            loss = loss + self.config.listwise_loss_weight * lw
        if self.config.listwise_topk_weight > 0 and "group_ids" in batch:
            # ADDED to the full-set term, never replacing it. Replacing it
            # measured 90.5 -> 89.5 on CASF: restricting the softmax to the top
            # 5 did win 10 of the 24 targets whose answer sat at rank 2-5, which
            # is what it was for, but it lost 13 others that the full-set term
            # had been ordering correctly -- with no gradient left for the far
            # poses, the global ranking decays. The two terms do different jobs.
            lwk = self._listwise_loss(
                pred, rmsd, batch["group_ids"], topk=self.config.listwise_topk
            )
            loss = loss + self.config.listwise_topk_weight * lwk
            self.log(f"{stage}/list", lw, prog_bar=True, sync_dist=sync)
        if self.config.mlm_aux_weight > 0:
            mlm = self._mlm_aux_loss(batch)
            loss = loss + self.config.mlm_aux_weight * mlm
            self.log(f"{stage}/mlm", mlm, prog_bar=True, sync_dist=sync)
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

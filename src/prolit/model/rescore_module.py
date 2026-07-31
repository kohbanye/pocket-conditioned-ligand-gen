"""LightningModule for the pose-scoring head (discriminative rescorer).

The pretrained complex-token MLM encoder (:class:`~prolit.model.complex_mlm.ComplexMLM`,
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

from prolit.model.complex_mlm import build_complex_mlm, count_parameters

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
        self.pooling: str = config.pooling
        # meanmax and xattn concatenate a second pooled vector with the ligand
        # mean, so the head sees 2H; mean/attn produce a single H vector.
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
        if self.pooling == "pairsum":
            # GenScore-style pairwise interaction: for every (ligand token,
            # pocket token) pair, score a learned interaction in each of C
            # channels and sum over pairs -- the explicit sum-over-contacts
            # inductive bias the pooled readouts lack. The C-dim interaction
            # vector is concatenated with the ligand mean.
            self.pair_heads: int = config.pair_heads
            self.pair_q = nn.Linear(h, h)
            self.pair_k = nn.Linear(h, h)
            in_dim = h + self.pair_heads
        # Optional trainable interaction transformer over the token states,
        # inserted before pooling. The MLM encoder is pretrained for masked-token
        # prediction, not affinity; these fresh layers give the head capacity to
        # re-model the pocket-ligand interface from the tokens (no tokenizer
        # change). Pooling-agnostic: refines hs, then _pool runs as usual.
        n_int = getattr(config, "head_interaction_layers", 0)
        if n_int > 0:
            layer = nn.TransformerEncoderLayer(
                h,
                nhead=12,
                dim_feedforward=4 * h,
                dropout=config.head_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.interaction: nn.TransformerEncoder | None = nn.TransformerEncoder(
                layer, num_layers=n_int
            )
        else:
            self.interaction: nn.TransformerEncoder | None = None
        # Dense auxiliary supervision: predict how far each ligand atom sits from
        # its native position. One RMSD scalar says only "this pose is wrong";
        # ~30 per-atom labels say WHICH atoms are wrong, which is the same
        # quantity the pose score aggregates and gives the interface a per-atom
        # error signal instead of a single pooled one.
        self.atom_head: nn.Linear | None = (
            nn.Linear(h, 1) if config.atom_aux_weight > 0 else None
        )
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
        self._logged: bool = False

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
        if self.pooling == "pairsum":
            pocket = batch["attention_mask"].bool() & ~lig  # (B,L)
            b, ln, h = hs.shape
            nh, hd = self.pair_heads, h // self.pair_heads
            q = self.pair_q(hs).view(b, ln, nh, hd)
            k = self.pair_k(hs).view(b, ln, nh, hd)
            # e[b,c,i,j] = <q_i, k_j> for channel c; sum over ligand-pocket pairs.
            e = torch.einsum("bihd,bjhd->bhij", q, k) / (hd**0.5)  # (B,C,L,L)
            pair = (lig.unsqueeze(-1) & pocket.unsqueeze(1)).unsqueeze(1)  # (B,1,L,L)
            e = e.masked_fill(~pair, 0.0)
            # Normalize by ligand-atom count, not pair count: dividing by the
            # total pair count dilutes the few real contacts across a large
            # pocket's many non-contacting pairs. Per ligand atom = total pocket
            # interaction each ligand atom sees, pocket size aside.
            nlig = lig.sum(dim=1, keepdim=True).clamp(min=1.0).to(e.dtype)  # (B,1)
            inter = e.sum(dim=(2, 3)) / nlig  # (B,C) interaction per ligand atom
            return torch.cat([mean, inter], dim=-1)
        msg = f"unknown pooling: {self.pooling}"
        raise ValueError(msg)

    def _encode_states(self, batch: dict[str, Tensor]) -> Tensor:
        hs = self.encoder.encode(batch["input_ids"], batch["attention_mask"])  # (B,L,H)
        if self.interaction is not None:
            hs = self.interaction(
                hs, src_key_padding_mask=~batch["attention_mask"].bool()
            )
        return hs

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        hs = self._encode_states(batch)
        return self.head(self._pool(hs, batch)).squeeze(-1)  # (B,) predicted label

    def forward_with_atoms(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor | None]:
        """Pose prediction plus the per-atom displacement prediction (or None)."""
        hs = self._encode_states(batch)
        pred = self.head(self._pool(hs, batch)).squeeze(-1)
        atom = None if self.atom_head is None else self.atom_head(hs).squeeze(-1)
        return pred, atom

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

    def _listwise_loss(self, pred: Tensor, rmsd: Tensor, groups: Tensor) -> Tensor:
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
        pred, atom = self.forward_with_atoms(batch)
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
        if atom is not None and "disp" in batch:
            dm = batch["disp_mask"]
            if bool(dm.any()):
                tgt = batch["disp"].clamp(max=self.config.rmsd_cap)
                aux = nn.functional.smooth_l1_loss(atom[dm], tgt[dm])
                loss = loss + self.config.atom_aux_weight * aux
                self.log(f"{stage}/atom", aux, prog_bar=True, sync_dist=sync)
        if self.config.listwise_loss_weight > 0 and "group_ids" in batch:
            lw = self._listwise_loss(pred, rmsd, batch["group_ids"])
            loss = loss + self.config.listwise_loss_weight * lw
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

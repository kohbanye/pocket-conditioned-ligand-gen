"""LightningModule for from-scratch training of the pocket-conditioned LM."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import lightning as L
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from prolit.data.clm_dataset import IGNORE_INDEX, PAD_SEGMENT
from prolit.model.clm import build_qwen3_lm, count_parameters
from prolit.tokenizers.lm_vocab import L_CLOSE_ID, L_OPEN_ID, NUM_SPECIAL

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

    from prolit.config import CLMTrainingConfig


def build_block_diagonal_mask(segment_ids: Tensor, dtype: torch.dtype) -> Tensor:
    """Build a 4D additive attention mask from packed-document segment ids.

    A query at position ``i`` may attend key ``j`` iff ``j <= i`` (causal) and
    both belong to the same document (``segment_ids[i] == segment_ids[j]``).
    Right-padding shares ``PAD_SEGMENT`` so padded queries still attend earlier
    padding (and themselves), avoiding fully-masked rows / NaNs.

    Args:
        segment_ids: ``(B, L)`` long; ``PAD_SEGMENT`` (-1) on padding.
        dtype: floating dtype for the additive mask.

    Returns:
        ``(B, 1, L, L)`` additive mask: ``0`` where attention is allowed and
        the dtype's most-negative value where it is masked.
    """
    b, length = segment_ids.shape
    device = segment_ids.device
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
    same = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)  # (B, L, L)
    allowed = same & causal.unsqueeze(0)
    mask = torch.zeros(b, 1, length, length, dtype=dtype, device=device)
    mask.masked_fill_(~allowed.unsqueeze(1), torch.finfo(dtype).min)
    return mask


#: A document with fewer ligand codes than this has too noisy a centroid to
#: be worth a regression target.
MIN_LIGAND_CODES = 3


def geometry_cross_entropy(
    logits: Tensor, labels: Tensor, geo_idx: Tensor, geo_w: Tensor
) -> Tensor:
    """Cross-entropy whose target is a Gaussian over geometric neighbours.

    Codebook positions get the smoothed target; the specials (``<p>``, ``</l>``,
    padding) keep the hard one, because "end the molecule here" has no
    neighbour that is nearly right.
    """
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    logp = nn.functional.log_softmax(flat_logits.float(), dim=-1)
    hard = nn.functional.nll_loss(logp, flat_labels.clamp(min=0), reduction="none")
    code = flat_labels - NUM_SPECIAL
    smooth = (
        (flat_labels != IGNORE_INDEX) & (code >= 0) & (code < geo_idx.shape[0])
    )
    if smooth.any():
        c = code[smooth]
        # geo_idx holds codebook indices; the vocabulary offsets them.
        picked = logp[smooth].gather(1, geo_idx[c] + NUM_SPECIAL)
        hard = hard.masked_scatter(smooth, -(geo_w[c] * picked).sum(-1))
    return hard.view(labels.shape)


def _geometry_targets(
    xyz: Tensor, tau: float, k: int
) -> tuple[Tensor, Tensor]:
    """Each code's ``k`` nearest codes in the mean-coordinate table, and a
    Gaussian weight over them normalised to sum to one.

    Codes the table never saw sit at the origin, which would make them every
    other unseen code's nearest neighbour. They are given a one-hot target on
    themselves instead, so an unseen code is supervised exactly as plain
    cross-entropy would.
    """
    n = xyz.shape[0]
    k = min(k, n)
    seen = xyz.abs().sum(-1) > 0
    d = torch.cdist(xyz, xyz)
    d[:, ~seen] = float("inf")
    idx = d.topk(k, largest=False).indices
    w = torch.exp(-d.gather(1, idx) ** 2 / (2.0 * tau * tau))
    w = w / w.sum(-1, keepdim=True).clamp(min=1e-12)
    # Unseen codes: fall back to one-hot on themselves.
    self_idx = torch.arange(n).unsqueeze(1).expand(-1, k)
    one_hot = torch.zeros(n, k)
    one_hot[:, 0] = 1.0
    idx = torch.where(seen.unsqueeze(1), idx, self_idx)
    w = torch.where(seen.unsqueeze(1), w, one_hot)
    return idx.to(torch.long), w.to(torch.float32)


class ProLITCLMModule(L.LightningModule):
    """Trains a dense Qwen3 LM on packed VQ-VAE token blocks (loss on all tokens)."""

    # Declared here, not only assigned in __init__: LightningModule inherits
    # nn.Module's __getattr__, which widens every attribute to `Tensor | Module`
    # and loses the concrete type. The buffers below are registered
    # conditionally, so without these the type checker sees a Module where the
    # code indexes a table or compares a float.
    geo_idx: Tensor
    geo_w: Tensor
    code_xyz: Tensor

    def __init__(self, config: CLMTrainingConfig) -> None:
        super().__init__()
        self.config: CLMTrainingConfig = config
        self.save_hyperparameters()
        self.model = build_qwen3_lm(config.model)
        self._logged_param_count: bool = False
        # Auxiliary centroid regression.
        #
        # The first ligand token is predicted from the hidden state at ``<l>``.
        # Measured, the model's placement of that atom (2.13 A error with the
        # correct pocket) matches a predictor that never reads the pocket
        # (2.14 A), while a one-line geometric formula over the same pocket
        # reaches 1.83 A. So the pocket is legible and the ``<l>`` state does
        # not carry it. This head makes that state predict where the ligand
        # goes, which is the one thing the anchor needs from it.
        #
        # It lives outside ``self.model``, so a checkpoint trained without it
        # still loads (the head is simply absent and re-initialised).
        self.centroid_head: nn.Module | None = None
        if float(getattr(config, "centroid_loss_weight", 0.0)) > 0.0:
            hidden = config.model.hidden_size
            self.centroid_head = nn.Sequential(
                nn.Linear(hidden, hidden // 4), nn.GELU(),
                nn.Linear(hidden // 4, 3),
            )
            table = getattr(config, "code_mean_coords", "")
            if not table:
                msg = "centroid_loss_weight needs --code-mean-coords"
                raise ValueError(msg)
            t = torch.load(table, map_location="cpu")["table"]
            self.register_buffer("code_xyz", t, persistent=False)

        # Geometry-smoothed cross-entropy.
        self.geo_tau: float = float(getattr(config, "code_geometry_tau", 0.0))
        if self.geo_tau > 0.0:
            table = getattr(config, "code_mean_coords", "")
            if not table:
                msg = "code_geometry_tau needs --code-mean-coords"
                raise ValueError(msg)
            xyz = torch.load(table, map_location="cpu")["table"]
            k = int(getattr(config, "code_geometry_k", 32))
            idx, w = _geometry_targets(xyz, self.geo_tau, k)
            # Not persistent: the table is derived from the VQ-VAE and the
            # corpus, both of which the run records separately, and baking
            # 8192xK into every checkpoint would make them a third copy that
            # can silently disagree with the other two.
            self.register_buffer("geo_idx", idx, persistent=False)
            self.register_buffer("geo_w", w, persistent=False)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        # Build the block-diagonal causal mask in the parameter dtype so it
        # matches the attention scores' dtype under autocast.
        embed = cast("nn.Embedding", self.model.get_input_embeddings())
        attn_dtype = embed.weight.dtype
        attn_mask = build_block_diagonal_mask(batch["segment_ids"], attn_dtype)
        weight = float(getattr(self.config, "anchor_loss_weight", 1.0))
        if weight == 1.0 and self.centroid_head is None and self.geo_tau == 0.0:
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=attn_mask,
                position_ids=batch["position_ids"],
                labels=batch["labels"],
            )
            return out.loss
        # Up-weight the first few ligand tokens.
        #
        # The loss is a mean over ~25 ligand positions, so the atom that
        # anchors the molecule in the pocket carries 1/25 of it. Measured, that
        # is not enough for the model to learn to read the pocket at all: its
        # first-atom placement error with the correct pocket is 2.13 A, and a
        # predictor that ignores the pocket entirely and answers the mean
        # ligand centroid over all targets gets 2.14 A. The two are the same
        # number. Everything downstream inherits the anchor, so this is where
        # the gradient has to go.
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=attn_mask,
            position_ids=batch["position_ids"],
            output_hidden_states=self.centroid_head is not None,
        )
        logits = out.logits[:, :-1]
        labels = batch["labels"][:, 1:]
        per_token = (
            geometry_cross_entropy(logits, labels, self.geo_idx, self.geo_w)
            if self.geo_tau > 0.0
            else nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).view(labels.shape)
        )
        w = self._anchor_weights(batch["input_ids"][:, 1:], labels, weight)
        keep = labels != IGNORE_INDEX
        loss = (per_token * w)[keep].sum() / w[keep].sum()
        if self.centroid_head is not None:
            aux = self._centroid_loss(batch["input_ids"], out.hidden_states[-1])
            if aux is not None:
                self.log("train/centroid", aux.detach(), prog_bar=False)
                loss = loss + float(self.config.centroid_loss_weight) * aux
        return loss

    def _centroid_loss(self, ids: Tensor, hidden: Tensor) -> Tensor | None:
        """Predict each document's ligand centroid from its ``<l>`` state."""
        head = self.centroid_head
        if head is None:
            return None
        preds, targets = [], []
        opens = (ids == L_OPEN_ID).nonzero(as_tuple=False).tolist()
        closes = (ids == L_CLOSE_ID).nonzero(as_tuple=False)
        by_row: dict[int, list[int]] = {}
        for b, pos in closes.tolist():
            by_row.setdefault(b, []).append(pos)
        for b, pos in opens:
            ends = [q for q in by_row.get(b, ()) if q > pos]
            if not ends:
                continue
            codes = ids[b, pos + 1 : ends[0]] - NUM_SPECIAL
            codes = codes[(codes >= 0) & (codes < self.code_xyz.shape[0])]
            if codes.numel() < MIN_LIGAND_CODES:
                continue
            preds.append(hidden[b, pos])
            # The centroid is not what the anchor needs: the first atom sits a
            # median 1.98 A from it, so knowing the centroid to 0.6 A still
            # leaves the anchor on a 2 A shell. ``anchor`` targets the first
            # atom's own position instead.
            targets.append(
                self.code_xyz[codes[0]]
                if self.config.centroid_target == "anchor"
                else self.code_xyz[codes].mean(0)
            )
        if not preds:
            return None
        p = head(torch.stack(preds))
        t = torch.stack(targets).to(p.dtype)
        return nn.functional.smooth_l1_loss(p, t)

    def _anchor_weights(
        self, shifted_ids: Tensor, labels: Tensor, weight: float
    ) -> Tensor:
        """1.0 everywhere, ``weight`` on the first ``anchor_loss_atoms`` of each
        document's ligand block."""
        n = int(getattr(self.config, "anchor_loss_atoms", 3))
        w = torch.ones_like(labels, dtype=torch.float32)
        opens = shifted_ids == L_OPEN_ID
        idx = opens.nonzero(as_tuple=False)
        for b, pos in idx.tolist():
            w[b, pos + 1 : pos + 1 + n] = weight
        return w

    def on_fit_start(self) -> None:
        if not self._logged_param_count:
            n = count_parameters(self.model)
            self.print(f"Model parameters: {n / 1e6:.1f}M")
            self._logged_param_count: bool = True

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:  # noqa: ARG002
        loss = self(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/ppl", torch.exp(loss.detach()))
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:  # noqa: ARG002
        loss = self(batch)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/ppl", torch.exp(loss.detach()), sync_dist=True)

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:  # noqa: ARG002
        loss = self(batch)
        self.log("test/loss", loss, sync_dist=True)
        self.log("test/ppl", torch.exp(loss.detach()), sync_dist=True)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        # Decoupled weight decay: skip 1D params (norms, biases). Embeddings are
        # tied to the LM head, so decaying them is standard practice here.
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
            opt,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            opt,
            T_max=cosine_steps,
            eta_min=self.config.learning_rate * self.config.min_lr_ratio,
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


# Re-export so callers can build masks without importing the dataset module.
__all__ = ["PAD_SEGMENT", "ProLITCLMModule", "build_block_diagonal_mask"]

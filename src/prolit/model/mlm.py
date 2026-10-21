"""Self-implemented ESM3-style bidirectional MLM over complex tokens.

A from-scratch masked-language-model encoder, faithful to the transformer block
in Biohub/esm (ESM3): per-sublayer pre-LayerNorm, rotary attention with
QK-LayerNorm, a SwiGLU feed-forward whose hidden width is rounded to a multiple
of 256, residual connections scaled by ``sqrt(n_layers / 36)``, and a final
bias-free LayerNorm. The geometric/structure-token attention of ESM3 is dropped
-- our 3D information already lives in the VQ-VAE tokens, so this is the plain
(non-geometric) all-to-all encoder over one token track.

This is the representation backbone for pose rescoring: trained with a masked
objective over native complex tokens, a decoy pose gets a lower pseudo-likelihood
(or a fine-tuned head scores it). It is NOT HuggingFace's ``EsmModel``; only the
architecture is borrowed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from prolit.tokenizers.lm_vocab import (
    L_CLOSE_ID,
    L_OPEN_ID,
    P_CLOSE_ID,
    P_OPEN_ID,
    PAD_ID,
)

if TYPE_CHECKING:
    from prolit.config import ProLITMLMConfig

IGNORE_INDEX = -100


# --- rotary position embeddings (non-interleaved, ESM3 convention) ----------


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Caches cos/sin over positions and applies them per-head to q, k."""

    inv_freq: Tensor  # registered as a buffer, which a type checker cannot see

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len: int = 0
        self._cos: Tensor | None = None
        self._sin: Tensor | None = None

    def _update_cache(self, seq_len: int, device: torch.device) -> None:
        if (
            self._cos is not None
            and seq_len <= self._seq_len
            and self._cos.device == device
        ):
            return
        self._seq_len: int = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos = emb.cos()
        self._sin = emb.sin()

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        # q, k: (B, H, S, head_dim). Positions run along S.
        seq_len = q.shape[-2]
        self._update_cache(seq_len, q.device)
        cos = self._cos[:seq_len].to(q.dtype)  # type: ignore[index]
        sin = self._sin[:seq_len].to(q.dtype)  # type: ignore[index]
        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin
        return q_rot, k_rot


# --- attention / feed-forward (ESM3 UnifiedTransformerBlock) -----------------


class MultiHeadAttention(nn.Module):
    """LayerNorm -> QKV -> (QK-LayerNorm) -> rotary -> SDPA -> out proj."""

    def __init__(  # noqa: PLR0913
        self,
        d_model: int,
        n_heads: int,
        *,
        bias: bool,
        qk_layernorm: bool,
        rope_theta: float,
        dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.n_heads: int = n_heads
        self.d_head: int = d_model // n_heads
        self.dropout: float = dropout
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model, eps=layer_norm_eps),
            nn.Linear(d_model, d_model * 3, bias=bias),
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias, eps=layer_norm_eps)
            self.k_ln = nn.LayerNorm(d_model, bias=bias, eps=layer_norm_eps)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()
        self.rotary = RotaryEmbedding(self.d_head, base=rope_theta)

    def forward(self, x: Tensor, attn_bias: Tensor | None) -> Tensor:
        b, s, _ = x.shape
        qkv = self.layernorm_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k = self.q_ln(q), self.k_ln(k)

        def split(t: Tensor) -> Tensor:
            return t.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)  # (B, H, S, d_head)
        q, k = self.rotary(q, k)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(b, s, self.n_heads * self.d_head)
        return self.out_proj(out)


def swiglu_hidden(expansion_ratio: float, d_model: int) -> int:
    """Round the SwiGLU hidden width to the nearest multiple of 256 (ESM3)."""
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ffn(
    d_model: int, expansion_ratio: float, *, bias: bool, layer_norm_eps: float
) -> nn.Sequential:
    hidden = swiglu_hidden(expansion_ratio, d_model)
    return nn.Sequential(
        nn.LayerNorm(d_model, eps=layer_norm_eps),
        nn.Linear(d_model, hidden * 2, bias=bias),
        SwiGLU(),
        nn.Linear(hidden, d_model, bias=bias),
    )


class TransformerBlock(nn.Module):
    """ESM3 UnifiedTransformerBlock (plain attention, no geometric track)."""

    def __init__(self, cfg: ProLITMLMConfig, scaling_factor: float) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(
            cfg.hidden_size,
            cfg.num_attention_heads,
            bias=cfg.bias,
            qk_layernorm=cfg.qk_layernorm,
            rope_theta=cfg.rope_theta,
            dropout=cfg.dropout,
            layer_norm_eps=cfg.layer_norm_eps,
        )
        self.ffn = swiglu_ffn(
            cfg.hidden_size,
            cfg.ffn_expansion_ratio,
            bias=cfg.bias,
            layer_norm_eps=cfg.layer_norm_eps,
        )
        self.scaling_factor: float = scaling_factor

    def forward(self, x: Tensor, attn_bias: Tensor | None) -> Tensor:
        x = x + self.attn(x, attn_bias) / self.scaling_factor
        return x + self.ffn(x) / self.scaling_factor


# --- MLM head + full model ---------------------------------------------------


class MLMHead(nn.Module):
    """Dense -> GELU -> LayerNorm -> vocab projection (+ bias)."""

    def __init__(self, d_model: int, vocab_size: int, layer_norm_eps: float) -> None:
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x: Tensor) -> Tensor:
        x = F.gelu(self.dense(x))
        x = self.layer_norm(x)
        return self.decoder(x) + self.bias


@dataclass
class MLMOutput:
    """Mimics the HF masked-LM output so the LightningModule stays generic."""

    loss: Tensor | None
    logits: Tensor
    #: Auxiliary per-token contact losses, keyed by target name. Empty unless
    #: the heads are configured AND targets were passed.
    contact_losses: dict[str, Tensor] = field(default_factory=dict)
    contact_logits: dict[str, Tensor] = field(default_factory=dict)


def component_balanced_mlm_loss(
    logits: Tensor, labels: Tensor, input_ids: Tensor
) -> Tensor:
    """Mean component CE per example, then mean over nonempty examples.

    Intact boundary markers identify protein/ligand spans even after corruption.
    Missing components are omitted rather than halving a ligand-only objective.
    Tensor-only reductions keep empty ranks differentiable and DDP compatible.
    """
    ce = F.cross_entropy(
        logits.flatten(0, 1).float(),
        labels.flatten(),
        ignore_index=-100,
        reduction="none",
    ).view_as(labels)
    means, present = [], []
    for opening, closing in ((P_OPEN_ID, P_CLOSE_ID), (L_OPEN_ID, L_CLOSE_ID)):
        inside = (input_ids == opening).cumsum(-1) > (input_ids == closing).cumsum(-1)
        selected = inside & (labels != IGNORE_INDEX)
        count = selected.sum(-1)
        means.append((ce * selected).sum(-1) / count.clamp_min(1))
        present.append(count > 0)
    counts = torch.stack(present, -1).sum(-1)
    per_example = torch.stack(means, -1).sum(-1) / counts.clamp_min(1)
    return per_example.sum() / (counts > 0).sum().clamp_min(1)


class ProLITMLM(nn.Module):
    """ESM3-style encoder + MLM head over the VQ-VAE complex-token vocabulary."""

    def __init__(self, cfg: ProLITMLMConfig) -> None:
        super().__init__()
        self.cfg: ProLITMLMConfig = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=PAD_ID)
        scaling = math.sqrt(cfg.num_hidden_layers / 36) if cfg.scale_residue else 1.0
        self.layers = nn.ModuleList(
            TransformerBlock(cfg, scaling) for _ in range(cfg.num_hidden_layers)
        )
        self.norm = nn.LayerNorm(cfg.hidden_size, bias=False, eps=cfg.layer_norm_eps)
        self.head = MLMHead(cfg.hidden_size, cfg.vocab_size, cfg.layer_norm_eps)
        # getattr: an older pickled config has no such field, and building a
        # model from it must not raise.
        self.contact_heads = nn.ModuleDict()
        if getattr(cfg, "contact_heads", False):
            self.contact_heads["res"] = nn.Linear(
                cfg.hidden_size, getattr(cfg, "contact_residue_classes", 21)
            )
            self.contact_heads["itype"] = nn.Linear(
                cfg.hidden_size, getattr(cfg, "contact_itype_classes", 4)
            )
        if cfg.tie_word_embeddings:
            self.head.decoder.weight = self.embed.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def encode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        return_all_layers: bool = False,
    ) -> Tensor:
        """Return the final normed hidden states (B, L, hidden), pre-MLM-head.

        Shared by the MLM objective and by the pose-scoring head, which pools
        these representations over the ligand span instead of predicting tokens.

        ``return_all_layers`` instead returns ``(1 + num_hidden_layers, B, L,
        hidden)``: the embedding output followed by every layer's output, each
        through this module's own final LayerNorm so the states share one scale.
        It adds no parameters, and the last state is what the single-tensor form
        returns, so a head that mixes the stack can reproduce today's behaviour
        exactly by putting all its weight on the last state.
        """
        # Additive key-padding bias (B, 1, 1, S): 0 for real tokens, large
        # negative for pad. Built with masked_fill to avoid 0 * -inf = NaN.
        attn_bias = None
        if attention_mask is not None:
            dtype = self.embed.weight.dtype
            pad = attention_mask[:, None, None, :] == 0
            attn_bias = torch.zeros(pad.shape, dtype=dtype, device=input_ids.device)
            attn_bias.masked_fill_(pad, torch.finfo(dtype).min)

        x = self.embed(input_ids)
        if not return_all_layers:
            for layer in self.layers:
                x = layer(x, attn_bias)
            return self.norm(x)
        states = [self.norm(x)]
        for layer in self.layers:
            x = layer(x, attn_bias)
            states.append(self.norm(x))
        return torch.stack(states, dim=0)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        *,
        component_balanced_loss: bool = False,
        contact_targets: dict[str, Tensor] | None = None,
    ) -> MLMOutput:
        x = self.encode(input_ids, attention_mask)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = (
                component_balanced_mlm_loss(logits, labels, input_ids)
                if component_balanced_loss
                else F.cross_entropy(
                    logits.view(-1, logits.size(-1)).float(),
                    labels.view(-1),
                    ignore_index=-100,
                )
            )
        contact_losses: dict[str, Tensor] = {}
        contact_logits: dict[str, Tensor] = {}
        for name, head in self.contact_heads.items():
            aux_logits = head(x)
            contact_logits[name] = aux_logits
            target = (contact_targets or {}).get(name)
            if target is None:
                continue
            # A batch can legitimately carry no labelled position -- a
            # ligand-only document has no contact to predict -- and
            # cross_entropy over an all-ignored target returns NaN, which would
            # poison the total. Skip instead.
            if not bool((target != IGNORE_INDEX).any()):
                continue
            contact_losses[name] = F.cross_entropy(
                aux_logits.reshape(-1, aux_logits.size(-1)).float(),
                target.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        return MLMOutput(
            loss=loss,
            logits=logits,
            contact_losses=contact_losses,
            contact_logits=contact_logits,
        )


def build_prolit_mlm(cfg: ProLITMLMConfig) -> ProLITMLM:
    """Construct a randomly-initialized ESM3-style complex-token MLM."""
    return ProLITMLM(cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

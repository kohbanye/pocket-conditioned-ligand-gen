"""LoRA adapters for the frozen MLM encoder's attention projections.

Pre-registered in docs/notes/2026-09-15_prereg_lora.md. The encoder's own
weights are never updated; each wrapped projection gains a rank-r pair whose
second matrix starts at zero, so a freshly wrapped encoder is bit-identical to
the frozen one and only training can move it. That property is what separates
this from the full fine-tuning a dev diagnostic already refuted: the adaptation
starts as a no-op and adds a few hundred thousand parameters, not 99M.
"""

from __future__ import annotations

from torch import Tensor, nn

#: Attention submodule attributes that get an adapter: the fused QKV projection
#: (a Linear inside layernorm_qkv) and the output projection.
LORA_INIT_STD = 0.02


class LoRALinear(nn.Module):
    """``base(x) + (alpha / rank) * B(A(x))`` with ``B`` initialised to zeros."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            msg = f"rank must be positive, got {rank}"
            raise ValueError(msg)
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.normal_(self.lora_a.weight, std=LORA_INIT_STD)
        nn.init.zeros_(self.lora_b.weight)
        self.scaling: float = alpha / rank

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scaling


def apply_lora(encoder: nn.Module, rank: int, alpha: float) -> int:
    """Wrap every attention block's QKV and output projections. Returns the count.

    Called AFTER the MLM checkpoint is loaded, so the load still matches the
    plain encoder's state_dict; wrapping first would rename every key.
    """
    wrapped = 0
    # nn.Module.__getattr__ is typed `Tensor | Module`, so the checker cannot see
    # that `.layers` is a ModuleList or that a block has `.attn`. Both hold for
    # every encoder this is called on, and tests/test_lora.py asserts the
    # wrapping actually happened.
    for block in encoder.layers:  # ty: ignore[not-iterable]
        attn = block.attn  # ty: ignore[possibly-missing-attribute]
        attn.layernorm_qkv[1] = LoRALinear(attn.layernorm_qkv[1], rank, alpha)
        attn.out_proj = LoRALinear(attn.out_proj, rank, alpha)
        wrapped += 2
    if not wrapped:
        msg = "no attention blocks found to adapt"
        raise ValueError(msg)
    return wrapped

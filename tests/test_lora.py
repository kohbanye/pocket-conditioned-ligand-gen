"""LoRA adapters start as a no-op and are the only encoder parameters that train.

Pre-registered in docs/notes/2026-09-15_prereg_lora.md. If the zero-initialised
second matrix ever stops making a freshly wrapped projection identical to the
frozen one, the arm is training a different model than the one it claims to.
"""

from __future__ import annotations

import torch
from torch import nn

from prolit.config import RescoreTrainingConfig
from prolit.model.lora import LoRALinear, apply_lora
from prolit.model.rescore_module import ComplexRescoreModule

LAYERS = 2
HIDDEN = 32
RANK = 4
ADAPTERS_PER_BLOCK = 2


def _module(rank: int) -> ComplexRescoreModule:
    cfg = RescoreTrainingConfig()
    cfg.model.num_hidden_layers = LAYERS
    cfg.model.hidden_size = HIDDEN
    cfg.model.num_attention_heads = 2
    cfg.lora_rank = rank
    # The regime the arm runs in: the encoder is frozen and only the adapters
    # and the head train. Without this the default config leaves all ~105M
    # encoder parameters trainable, which is what the control arm does.
    cfg.freeze_encoder = True
    return ComplexRescoreModule(cfg, lora_rank=rank).eval()


def test_wrapped_projection_starts_identical() -> None:
    base = nn.Linear(HIDDEN, HIDDEN)
    wrapped = LoRALinear(base, RANK, 1.0)
    x = torch.randn(3, HIDDEN)
    with torch.no_grad():
        torch.testing.assert_close(wrapped(x), base(x))


def test_a_nonzero_b_zero_at_init() -> None:
    wrapped = LoRALinear(nn.Linear(HIDDEN, HIDDEN), RANK, 1.0)
    assert float(wrapped.lora_a.weight.abs().sum()) > 0.0
    assert float(wrapped.lora_b.weight.abs().sum()) == 0.0


def test_adapters_exist_only_when_requested() -> None:
    assert [n for n, _ in _module(0).named_parameters() if "lora_" in n] == []
    names = [n for n, _ in _module(RANK).named_parameters() if "lora_" in n]
    assert len(names) == LAYERS * ADAPTERS_PER_BLOCK * 2


def test_only_adapters_and_head_train() -> None:
    module = _module(RANK)
    trainable = [n for n, p in module.named_parameters() if p.requires_grad]
    assert trainable, "nothing is trainable"
    for name in trainable:
        assert "lora_" in name or name.startswith("head."), name


def test_apply_lora_counts_every_projection() -> None:
    module = _module(0)
    wrapped = apply_lora(module.encoder, RANK, 1.0)
    assert wrapped == LAYERS * ADAPTERS_PER_BLOCK

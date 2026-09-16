"""The head can read a learned mix over every encoder layer, not just the last.

Pre-registered in docs/notes/2026-09-15_prereg_layer_mix.md. The invariants that
matter: the option adds parameters ONLY when asked for (so existing checkpoints
keep their state_dict), and putting all the mixing weight on the last state
reproduces exactly what every head before this read.
"""

from __future__ import annotations

import torch

from prolit.config import RescoreTrainingConfig
from prolit.model.rescore_module import ComplexRescoreModule

LAYERS = 2
HIDDEN = 32
BATCH, SEQ = 2, 7
ONE_HOT_LOGIT = 50.0


def _module(*, layer_mix: bool) -> ComplexRescoreModule:
    cfg = RescoreTrainingConfig()
    cfg.model.num_hidden_layers = LAYERS
    cfg.model.hidden_size = HIDDEN
    cfg.model.num_attention_heads = 2
    return ComplexRescoreModule(cfg, layer_mix=layer_mix).eval()


def _batch(module: ComplexRescoreModule) -> dict[str, torch.Tensor]:
    vocab = module.encoder.embed.num_embeddings
    gen = torch.Generator().manual_seed(7)
    ids = torch.randint(10, vocab, (BATCH, SEQ), generator=gen)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def test_weights_exist_only_when_requested() -> None:
    plain = _module(layer_mix=False)
    off = [n for n, _ in plain.named_parameters() if "layer_weights" in n]
    assert off == []
    on = _module(layer_mix=True)
    named = dict(on.named_parameters())
    assert list(named["layer_weights"].shape) == [LAYERS + 1]


def test_all_layers_last_state_is_the_plain_encode() -> None:
    module = _module(layer_mix=True)
    batch = _batch(module)
    with torch.no_grad():
        stack = module.encoder.encode(
            batch["input_ids"], batch["attention_mask"], return_all_layers=True
        )
        plain = module.encoder.encode(batch["input_ids"], batch["attention_mask"])
    assert stack.shape[0] == LAYERS + 1
    torch.testing.assert_close(stack[-1], plain)


def test_one_hot_mix_reproduces_the_last_layer() -> None:
    module = _module(layer_mix=True)
    batch = _batch(module)
    with torch.no_grad():
        module.layer_weights.copy_(
            torch.tensor([0.0] * LAYERS + [ONE_HOT_LOGIT])
        )
        mixed = module._encode_states(batch)  # noqa: SLF001
        plain = module.encoder.encode(batch["input_ids"], batch["attention_mask"])
    torch.testing.assert_close(mixed, plain, atol=1e-5, rtol=1e-4)


def test_zero_weights_average_every_state() -> None:
    module = _module(layer_mix=True)
    batch = _batch(module)
    with torch.no_grad():
        mixed = module._encode_states(batch)  # noqa: SLF001
        stack = module.encoder.encode(
            batch["input_ids"], batch["attention_mask"], return_all_layers=True
        )
    torch.testing.assert_close(mixed, stack.mean(dim=0), atol=1e-6, rtol=1e-5)


def test_plain_module_is_unchanged() -> None:
    module = _module(layer_mix=False)
    batch = _batch(module)
    with torch.no_grad():
        states = module._encode_states(batch)  # noqa: SLF001
        plain = module.encoder.encode(batch["input_ids"], batch["attention_mask"])
    torch.testing.assert_close(states, plain)


INIT_LAST = 4.0
LR_SCALE = 100.0
EXPECTED_LAST_WEIGHT = 0.8


def test_init_last_starts_the_mix_on_the_last_layer() -> None:
    cfg = RescoreTrainingConfig()
    cfg.model.num_hidden_layers = LAYERS
    cfg.model.hidden_size = HIDDEN
    cfg.model.num_attention_heads = 2
    cfg.layer_mix_init_last = INIT_LAST
    module = ComplexRescoreModule(cfg, layer_mix=True)
    weights = torch.softmax(module.layer_weights.detach(), dim=0)
    assert float(weights[-1]) > EXPECTED_LAST_WEIGHT
    assert float(weights[:-1].max()) < 1.0 - EXPECTED_LAST_WEIGHT


def test_mixing_weights_get_their_own_learning_rate() -> None:
    cfg = RescoreTrainingConfig()
    cfg.model.num_hidden_layers = LAYERS
    cfg.model.hidden_size = HIDDEN
    cfg.model.num_attention_heads = 2
    cfg.layer_mix_lr_scale = LR_SCALE
    module = ComplexRescoreModule(cfg, layer_mix=True)
    groups = module._param_groups()  # noqa: SLF001
    scaled = [g for g in groups if "lr" in g]
    assert len(scaled) == 1
    assert scaled[0]["params"] == [module.layer_weights]
    assert scaled[0]["lr"] == cfg.learning_rate * LR_SCALE
    others = [p for g in groups if "lr" not in g for p in g["params"]]
    assert all(p is not module.layer_weights for p in others)


def test_plain_head_has_no_scaled_group() -> None:
    groups = _module(layer_mix=False)._param_groups()  # noqa: SLF001
    assert all("lr" not in g for g in groups)

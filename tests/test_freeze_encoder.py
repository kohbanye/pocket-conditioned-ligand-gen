"""Freezing must survive Lightning putting the module back into train mode.

``nn.Module.train`` recurses into every child, and Lightning calls it on the
whole module at the start of each epoch. Without an override the encoder is
returned to train mode every epoch and resumes applying dropout: the weights
still do not update, so the parameter count and the gradients look right, and
the only symptom is that the head is fitting a moving target.
"""

from __future__ import annotations

from prolit.config import ProLITMLMConfig, RescoreTrainingConfig
from prolit.model.rescore_module import ComplexRescoreModule


def _module(*, freeze: bool) -> ComplexRescoreModule:
    config = RescoreTrainingConfig(model=ProLITMLMConfig(atom_codebook_size=64))
    config.freeze_encoder = freeze
    return ComplexRescoreModule(config)


def _trainable(module: ComplexRescoreModule) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def test_freezing_leaves_only_the_head_trainable() -> None:
    frozen = _module(freeze=True)
    head = sum(p.numel() for p in frozen.head.parameters())
    assert _trainable(frozen) == head
    assert all(not p.requires_grad for p in frozen.encoder.parameters())


def test_unfrozen_is_the_default_and_trains_everything() -> None:
    full = _module(freeze=False)
    assert RescoreTrainingConfig().freeze_encoder is False
    assert _trainable(full) > sum(p.numel() for p in full.head.parameters())


def test_encoder_stays_in_eval_mode_after_lightning_calls_train() -> None:
    frozen = _module(freeze=True)
    frozen.train()
    assert not frozen.encoder.training, "dropout would be active again"
    assert frozen.head.training, "the head must still train"


def test_an_unfrozen_encoder_still_follows_train_mode() -> None:
    full = _module(freeze=False)
    full.train()
    assert full.encoder.training
    full.eval()
    assert not full.encoder.training

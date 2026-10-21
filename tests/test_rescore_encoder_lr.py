"""Discriminative fine-tuning keeps LR ratios without altering the legacy arm."""

from types import SimpleNamespace

import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from prolit.config import ProLITMLMConfig, RescoreTrainingConfig
from prolit.model.rescore_module import ComplexRescoreModule


def _module(scale: float = 1.0, *, freeze: bool = False) -> ComplexRescoreModule:
    cfg = RescoreTrainingConfig(
        model=ProLITMLMConfig(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            atom_codebook_size=16,
        ),
        encoder_lr_scale=scale,
        learning_rate=1e-3,
        min_lr_ratio=0.1,
        warmup_steps=2,
        freeze_encoder=freeze,
    )
    model = ComplexRescoreModule(cfg)
    model.trainer = SimpleNamespace(estimated_stepping_batches=10)
    return model


@pytest.mark.parametrize("scale", [0.1, 0.3])
@pytest.mark.parametrize("freeze", [False, True])
def test_scaled_groups_cover_parameters_and_preserve_schedule(
    scale: float, freeze: bool  # noqa: FBT001
) -> None:
    model = _module(scale, freeze=freeze)
    result = model.configure_optimizers()
    opt, sched = result["optimizer"], result["lr_scheduler"]["scheduler"]
    ids = [id(p) for group in opt.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    encoder = {id(p) for p in model.encoder.parameters()}
    for step in range(11):
        for group in opt.param_groups:
            flags = {id(p) in encoder for p in group["params"]}
            assert len(flags) == 1
            base = 1e-3 * (scale if True in flags else 1.0)
            assert group["weight_decay"] == (
                model.config.weight_decay if group["params"][0].ndim >= 2 else 0.0
            )
            if step in (0, 2, 10):
                assert group["lr"] == pytest.approx(
                    base * {0: 0.001, 2: 1.0, 10: 0.1}[step]
                )
        normalized = [group["lr"] / group["initial_lr"] for group in opt.param_groups]
        assert max(normalized) == pytest.approx(min(normalized))
        opt.step()
        sched.step()


def test_default_optimizer_and_every_schedule_step_match_legacy() -> None:
    model = _module()
    result = model.configure_optimizers()
    actual = result["optimizer"]
    decay, no_decay = [], []
    for param in model.parameters():
        (decay if param.ndim >= 2 else no_decay).append(param)
    legacy = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": model.config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=model.config.learning_rate,
        betas=(model.config.adam_beta1, model.config.adam_beta2),
    )
    legacy_sched = SequentialLR(
        legacy,
        schedulers=[
            LinearLR(legacy, start_factor=1e-3, end_factor=1.0, total_iters=2),
            CosineAnnealingLR(legacy, T_max=8, eta_min=1e-4),
        ],
        milestones=[2],
    )
    assert isinstance(result["lr_scheduler"]["scheduler"], SequentialLR)
    assert actual.defaults == legacy.defaults
    for left, right in zip(actual.param_groups, legacy.param_groups, strict=True):
        assert [id(p) for p in left["params"]] == [id(p) for p in right["params"]]
    for _ in range(11):
        assert [g["lr"] for g in actual.param_groups] == [
            g["lr"] for g in legacy.param_groups
        ]
        actual.step()
        legacy.step()
        result["lr_scheduler"]["scheduler"].step()
        legacy_sched.step()


@pytest.mark.parametrize("scale", [0.0, -0.1, 1.01, float("nan")])
def test_reject_invalid_encoder_lr_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="encoder_lr_scale"):
        _module(scale)

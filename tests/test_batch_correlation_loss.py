"""The correlation term is scale-free, which is why it is worth adding.

Squared error is minimised by shrinking predictions toward the mean whenever
the signal is uncertain, and the CASF diagnosis measured exactly that (target
level spread ~19% too small). ``1 - Pearson`` does not move under an affine
rescaling, so shrinking buys nothing against it.
"""

from __future__ import annotations

import pytest
import torch

from prolit.model.rescore_module import batch_correlation_loss


def test_perfect_correlation_is_zero() -> None:
    target = torch.tensor([3.0, 5.0, 7.0, 11.0])
    same = batch_correlation_loss(target.clone(), target)
    assert same == pytest.approx(0.0, abs=1e-6)


def test_anticorrelation_is_two() -> None:
    target = torch.tensor([3.0, 5.0, 7.0, 11.0])
    assert batch_correlation_loss(-target, target) == pytest.approx(2.0, abs=1e-6)


def test_shrinking_toward_the_mean_does_not_change_it() -> None:
    target = torch.tensor([3.0, 5.0, 7.0, 11.0])
    pred = torch.tensor([3.5, 4.8, 7.4, 10.0])
    shrunk = pred.mean() + 0.2 * (pred - pred.mean())  # 5x compression
    assert batch_correlation_loss(shrunk, target) == pytest.approx(
        float(batch_correlation_loss(pred, target)), abs=1e-6
    )
    # ... while smooth-L1 rewards the shrunk prediction far less.
    l1 = torch.nn.functional.smooth_l1_loss
    assert l1(shrunk, target) > l1(pred, target)


def test_constant_prediction_is_zero_not_nan() -> None:
    target = torch.tensor([3.0, 5.0, 7.0, 11.0])
    out = batch_correlation_loss(torch.full_like(target, 4.0), target)
    assert torch.isfinite(out)
    assert out == pytest.approx(0.0)


def test_gradient_flows() -> None:
    pred = torch.tensor([1.0, 2.0, 3.0, 4.5], requires_grad=True)
    target = torch.tensor([2.0, 1.0, 4.0, 3.0])
    batch_correlation_loss(pred, target).backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0

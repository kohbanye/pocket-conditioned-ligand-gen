"""The group-mean term penalises a uniformly offset group, which per-complex
regression cannot distinguish from scattered error of the same size.

That distinction is the whole point: on CASF the head's target-level
differences come out ~19% too small, and a per-complex loss charges the same
for both failure shapes.
"""

from __future__ import annotations

import pytest
import torch

from prolit.model.rescore_module import group_mean_loss


def test_offset_group_is_penalised_and_scattered_group_is_not() -> None:
    target = torch.tensor([5.0, 6.0, 7.0, 8.0])
    groups = torch.tensor([0, 0, 1, 1])
    # Same per-complex error size in both cases: every residual is +/-0.5.
    offset = target + torch.tensor([0.5, 0.5, 0.5, 0.5])
    scattered = target + torch.tensor([0.5, -0.5, -0.5, 0.5])
    assert group_mean_loss(offset, target, groups) > group_mean_loss(
        scattered, target, groups
    )
    assert group_mean_loss(scattered, target, groups) == pytest.approx(0.0, abs=1e-6)


def test_matches_smooth_l1_of_the_group_means() -> None:
    pred = torch.tensor([1.0, 3.0, 10.0])
    target = torch.tensor([2.0, 2.0, 9.0])
    groups = torch.tensor([7, 7, 9])  # ids need not be 0..k-1
    expected = torch.nn.functional.smooth_l1_loss(
        torch.tensor([2.0, 10.0]), torch.tensor([2.0, 9.0])
    )
    assert group_mean_loss(pred, target, groups) == pytest.approx(float(expected))


def test_singleton_groups_reduce_to_the_per_complex_term() -> None:
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 2.0])
    groups = torch.tensor([0, 1, 2])
    expected = torch.nn.functional.smooth_l1_loss(pred, target)
    assert group_mean_loss(pred, target, groups) == pytest.approx(float(expected))


def test_gradient_flows_to_the_prediction() -> None:
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    target = torch.tensor([2.0, 3.0, 3.0, 4.0])
    groups = torch.tensor([0, 0, 1, 1])
    group_mean_loss(pred, target, groups).backward()
    assert pred.grad is not None
    # The offset group (0) carries gradient; the matched group (1) does not.
    assert pred.grad[:2].abs().sum() > 0
    assert pred.grad[2:].abs().sum() == pytest.approx(0.0, abs=1e-6)

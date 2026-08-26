"""Geometry-smoothed cross-entropy: a near miss must cost less than a far one.

Plain cross-entropy over a codebook cannot tell those apart, which is the whole
point of the smoothing, so the test is exactly that comparison.
"""

from __future__ import annotations

import torch

from prolit.data.clm_dataset import IGNORE_INDEX
from prolit.model.clm_module import _geometry_targets, geometry_cross_entropy
from prolit.tokenizers.lm_vocab import NUM_SPECIAL


def _line_table(n: int = 8) -> torch.Tensor:
    """Codes on a line 1 A apart, so "geometrically near" is unambiguous."""
    t = torch.zeros(n, 3)
    t[:, 0] = torch.arange(1, n + 1, dtype=torch.float32)
    return t


def test_weights_sum_to_one_and_peak_on_the_true_code() -> None:
    idx, w = _geometry_targets(_line_table(), tau=1.0, k=4)
    assert torch.allclose(w.sum(-1), torch.ones(8), atol=1e-5)
    # The nearest code to any code is itself, at distance zero.
    assert (idx[:, 0] == torch.arange(8)).all()
    assert (w[:, 0] == w.max(-1).values).all()


def test_unseen_codes_get_a_one_hot_target() -> None:
    t = _line_table()
    t[3] = 0.0  # a code the corpus never assigned
    idx, w = _geometry_targets(t, tau=1.0, k=4)
    assert idx[3, 0].item() == 3
    assert w[3, 0].item() == 1.0
    assert w[3, 1:].sum().item() == 0.0
    # ...and it is never anyone else's neighbour.
    assert (idx[torch.arange(8) != 3] != 3).all()


def test_a_near_miss_costs_less_than_a_far_one() -> None:
    idx, w = _geometry_targets(_line_table(), tau=1.0, k=4)
    vocab = NUM_SPECIAL + 8
    labels = torch.tensor([[NUM_SPECIAL + 4]])

    def loss_when_confident_about(code: int) -> float:
        logits = torch.full((1, 1, vocab), -10.0)
        logits[0, 0, NUM_SPECIAL + code] = 10.0
        return float(geometry_cross_entropy(logits, labels, idx, w))

    right = loss_when_confident_about(4)
    near = loss_when_confident_about(5)
    far = loss_when_confident_about(0)
    assert right < near < far


def test_specials_and_ignored_positions_keep_the_hard_target() -> None:
    idx, w = _geometry_targets(_line_table(), tau=1.0, k=4)
    vocab = NUM_SPECIAL + 8
    logits = torch.zeros(1, 2, vocab)
    labels = torch.tensor([[1, IGNORE_INDEX]])
    out = geometry_cross_entropy(logits, labels, idx, w)
    uniform = torch.log(torch.tensor(float(vocab)))
    # A special: plain cross-entropy, so -log(1/vocab) under flat logits.
    assert torch.isclose(out[0, 0], uniform, atol=1e-4)
    # Ignored positions are masked out by the caller; the value is finite.
    assert torch.isfinite(out[0, 1])

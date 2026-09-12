"""Iterative refinement must only touch the ligand, and only what it doubts.

The causal model's codes are mostly right -- 47.8% exactly, and the true code is
in its top 10 for 90.4% of atoms. So a refiner that rewrites everything throws
away more than it fixes; the whole point is to re-decide the *least confident*
positions with the rest of the molecule visible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from torch import nn

from prolit.model.mlm_decode import (
    MIN_LIGAND_CODES,
    apply_rigid,
    cold_decode,
    reconcile,
    refine_codes,
    rigid_between,
)

if TYPE_CHECKING:
    from collections.abc import Callable

from prolit.tokenizers.lm_vocab import (
    L_CLOSE_ID,
    L_OPEN_ID,
    NUM_SPECIAL,
    P_OPEN_ID,
)

NC = 16
VOCAB = NUM_SPECIAL + NC + 1
MASK_ID = NUM_SPECIAL + NC


class _Stub(nn.Module):
    """Always predicts code ``target`` and records what it was shown."""

    def __init__(self, target: int = 3) -> None:
        super().__init__()
        self.target = target
        self.seen: list[np.ndarray] = []

    def forward(self, input_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:  # noqa: ANN001, ARG002
        self.seen.append(input_ids[0].cpu().numpy().copy())
        out = torch.full((1, input_ids.shape[1], VOCAB), -10.0)
        out[..., NUM_SPECIAL + self.target] = 10.0
        return out


def test_it_rewrites_only_the_fraction_it_is_asked_to() -> None:
    model = _Stub(target=3)
    codes = [7] * 12
    out = refine_codes(
        model, MASK_ID, [1, 2], codes, codebook_size=NC, rounds=1, frac=0.25
    )
    assert len(out) == len(codes)
    # 25% of 12 = 3 positions move to the stub's target, the rest are untouched.
    assert sum(c == 3 for c in out) == 3
    assert sum(c == 7 for c in out) == 9


def test_the_pocket_and_the_markers_are_never_masked() -> None:
    model = _Stub()
    refine_codes(
        model, MASK_ID, [1, 2, 3], [5] * 8, codebook_size=NC, rounds=2, frac=0.5
    )
    for ids in model.seen:
        assert ids[0:2].tolist() == [1, P_OPEN_ID] or ids[1] == P_OPEN_ID
        # Pocket codes and the block markers survive every round.
        assert MASK_ID not in ids[: np.flatnonzero(ids == L_OPEN_ID)[0] + 1]
        assert ids[-1] == L_CLOSE_ID


def test_a_fragment_is_returned_untouched() -> None:
    model = _Stub()
    short = [4] * (MIN_LIGAND_CODES - 1)
    assert refine_codes(model, MASK_ID, [1], short, codebook_size=NC) == short
    assert model.seen == []


def test_zero_rounds_or_zero_fraction_is_a_no_op() -> None:
    model = _Stub()
    codes = [6] * 10
    assert refine_codes(model, MASK_ID, [1], codes, codebook_size=NC, rounds=0) == codes
    assert refine_codes(model, MASK_ID, [1], codes, codebook_size=NC, frac=0.0) == codes
    assert model.seen == []


def test_more_rounds_converge_rather_than_drift() -> None:
    model = _Stub(target=3)
    codes = [7] * 8
    out = refine_codes(
        model, MASK_ID, [1], codes, codebook_size=NC, rounds=8, frac=0.5
    )
    # Once every position holds the stub's own argmax there is nothing left to
    # doubt, so the answer stops moving.
    assert set(out) == {3}


def test_align_removes_a_rigid_move_but_keeps_internal_geometry() -> None:
    rng = np.random.default_rng(0)
    original = rng.normal(0, 2, (10, 3))
    # A pure rigid move of the whole molecule, plus one atom genuinely edited.
    theta = 0.4
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0], [0, 0, 1.0]])
    refined = original @ rot.T + np.array([3.0, -1.0, 2.0])
    refined[7] += np.array([1.5, 0.0, 0.0])

    out = reconcile(original, refined, changed=[7], mode="align")
    # The nine untouched atoms come back where they started...
    keep = [i for i in range(10) if i != 7]
    assert np.abs(out[keep] - original[keep]).max() < 1e-6
    # ...and the edited atom keeps the displacement that was the point.
    assert np.linalg.norm(out[7] - original[7]) > 1.0


def test_splice_restores_the_untouched_atoms_exactly() -> None:
    rng = np.random.default_rng(1)
    original = rng.normal(0, 2, (12, 3))
    refined = original + rng.normal(0, 0.3, (12, 3))  # everything drifted
    out = reconcile(original, refined, changed=[2, 5], mode="splice")
    keep = [i for i in range(12) if i not in (2, 5)]
    assert np.allclose(out[keep], original[keep])
    assert not np.allclose(out[2], original[2])


def test_off_and_degenerate_cases_return_the_refined_pose() -> None:
    original = np.zeros((6, 3))
    refined = np.ones((6, 3))
    assert np.allclose(reconcile(original, refined, [1], mode="off"), refined)
    # Too few unchanged atoms to define a superposition.
    many = list(range(5))
    assert np.allclose(reconcile(original, refined, many, mode="align"), refined)


def test_late_first_rewrites_the_tail_not_the_head() -> None:
    """The clash accumulates along the decode order, so the tail is the target.

    Measured over 40 targets: the fraction of ligand atoms within 3.0 A of the
    protein climbs 11.4% -> 33.7% from the first tenth of the sequence to the
    last, while FLOWR is flat (7.8% -> 8.1%). Confidence masking spreads its
    edits over the whole molecule and left that slope untouched.
    """
    n = 20
    codes = list(range(n))
    out = refine_codes(
        _Stub(target=3),
        MASK_ID,
        protein_codes=[1, 2],
        ligand_codes=codes,
        codebook_size=NC,
        rounds=1,
        frac=0.25,
        order="late_first",
    )
    changed = [i for i, (a, b) in enumerate(zip(codes, out, strict=True)) if a != b]
    assert changed, "one round of late_first must rewrite something"
    assert min(changed) >= n - 5, f"late_first touched the head: {changed}"


def test_late_first_sweeps_backwards_over_rounds() -> None:
    """Successive rounds march the block towards the front, without wrapping off."""
    n = 20
    codes = [7] * n
    out = refine_codes(
        _Stub(target=3),
        MASK_ID,
        protein_codes=[1],
        ligand_codes=codes,
        codebook_size=NC,
        rounds=2,
        frac=0.25,
        order="late_first",
    )
    changed = {i for i, c in enumerate(out) if c != 7}
    assert changed == set(range(n - 10, n)), (
        f"expected the last 10, got {sorted(changed)}"
    )


def test_late_first_never_leaves_the_ligand() -> None:
    """Even sweeping past the front, the block must stay inside the ligand."""
    stub = _Stub(target=3)
    n = MIN_LIGAND_CODES + 1
    refine_codes(
        stub,
        MASK_ID,
        protein_codes=[1, 2, 3],
        ligand_codes=list(range(n)),
        codebook_size=NC,
        rounds=12,
        frac=0.5,
        order="late_first",
    )
    for ids in stub.seen:
        assert ids[0] != MASK_ID
        assert P_OPEN_ID not in ids[ids == MASK_ID]
        masked = np.flatnonzero(ids == MASK_ID)
        lo = int(np.flatnonzero(ids == L_OPEN_ID)[0]) + 1
        hi = int(np.flatnonzero(ids == L_CLOSE_ID)[0])
        assert masked.size == 0 or (masked.min() >= lo and masked.max() < hi)


def test_cold_decode_fills_every_position() -> None:
    """A cold start commits every ligand position and leaves nothing masked."""
    out = cold_decode(
        _Stub(target=5),
        MASK_ID,
        protein_codes=[1, 2],
        n_ligand=17,
        codebook_size=NC,
        rounds=6,
        temperature=0.0,
    )
    assert len(out) == 17
    assert all(c == 5 for c in out), out
    assert MASK_ID not in out


def test_cold_decode_commits_gradually() -> None:
    """Early rounds must leave most positions masked (the cosine schedule)."""
    stub = _Stub(target=5)
    cold_decode(
        stub,
        MASK_ID,
        protein_codes=[1],
        n_ligand=20,
        codebook_size=NC,
        rounds=5,
        temperature=0.0,
    )
    masked = [int((ids == MASK_ID).sum()) for ids in stub.seen]
    assert masked[0] == 20, f"first round must see everything masked: {masked}"
    assert masked == sorted(masked, reverse=True), f"not monotone: {masked}"
    assert masked[-1] < masked[0]


def test_cold_decode_never_masks_the_pocket() -> None:
    stub = _Stub(target=2)
    cold_decode(
        stub,
        MASK_ID,
        protein_codes=[1, 2, 3, 4],
        n_ligand=12,
        codebook_size=NC,
        rounds=4,
        temperature=0.0,
    )
    for ids in stub.seen:
        lo = int(np.flatnonzero(ids == L_OPEN_ID)[0]) + 1
        hi = int(np.flatnonzero(ids == L_CLOSE_ID)[0])
        masked = np.flatnonzero(ids == MASK_ID)
        assert masked.min() >= lo
        assert masked.max() < hi


def test_cold_decode_refuses_a_degenerate_length() -> None:
    assert cold_decode(
        _Stub(), MASK_ID, protein_codes=[1], n_ligand=MIN_LIGAND_CODES - 1,
        codebook_size=NC, rounds=4,
    ) == []


class _Ranked(nn.Module):
    """Predicts a fixed preference order over codes, highest first."""

    def __init__(self, order: list[int]) -> None:
        super().__init__()
        self.order = order

    def forward(self, input_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:  # noqa: ANN001, ARG002
        out = torch.full((1, input_ids.shape[1], VOCAB), -10.0)
        for rank, code in enumerate(self.order):
            out[..., NUM_SPECIAL + code] = 10.0 - rank
        return out


def _probe(
    bad_positions: set[int], bad_codes: set[int]
) -> Callable[..., np.ndarray]:
    """A clash probe: a position is in the wall if it holds a bad code."""

    def fn(seqs: np.ndarray, only: int | None = None) -> np.ndarray:
        out = np.zeros(seqs.shape, dtype=bool)
        for p in bad_positions:
            if only is None or p == only:
                out[:, p] = np.isin(seqs[:, p], list(bad_codes))
        return out

    return fn


def test_clash_order_needs_a_probe() -> None:
    """Geometry does not live in this layer, so it must be handed in."""
    with pytest.raises(ValueError, match="clash_probe"):
        refine_codes(
            _Stub(), MASK_ID, [1], [5] * 8, codebook_size=NC, order="clash"
        )


def test_clash_order_rewrites_exactly_the_clashing_positions() -> None:
    codes = [7] * 12
    out = refine_codes(
        _Ranked([3, 4, 5]), MASK_ID, [1, 2], codes, codebook_size=NC,
        rounds=1, frac=1.0, order="clash",
        clash_probe=_probe({2, 9}, {7}),
    )
    assert out[2] != 7
    assert out[9] != 7
    assert [c for i, c in enumerate(out) if i not in (2, 9)] == [7] * 10


def test_clash_order_stops_when_nothing_is_in_the_wall() -> None:
    """A clean molecule must come back byte-identical, not re-decided."""
    codes = [7] * 12
    out = refine_codes(
        _Stub(target=3), MASK_ID, [1, 2], codes, codebook_size=NC,
        rounds=8, frac=1.0, order="clash",
        clash_probe=_probe(set(), {7}),
    )
    assert out == codes


def test_it_picks_the_most_probable_code_that_clears_the_wall() -> None:
    """Not the argmax, and not just any clear code -- the best clear one."""
    out = refine_codes(
        _Ranked([3, 4, 5]), MASK_ID, [1, 2], [7] * 10, codebook_size=NC,
        rounds=1, frac=1.0, order="clash",
        # 3 is the model's favourite but still clashes; 4 is next and is clear.
        clash_probe=_probe({0}, {7, 3}),
    )
    assert out[0] == 4


def test_it_falls_back_to_the_argmax_when_every_candidate_clashes() -> None:
    """A buried position degrades to ordinary MaskGIT rather than blocking."""
    out = refine_codes(
        _Ranked([3, 4, 5]), MASK_ID, [1, 2], [7] * 10, codebook_size=NC,
        rounds=1, frac=1.0, order="clash",
        clash_probe=_probe({0}, set(range(NC))),
    )
    assert out[0] == 3


def test_candidates_bounds_how_far_down_the_tail_it_will_reach() -> None:
    """With candidates=1 only the argmax is considered, so a clash stands."""
    out = refine_codes(
        _Ranked([3, 4, 5]), MASK_ID, [1, 2], [7] * 10, codebook_size=NC,
        rounds=1, frac=1.0, order="clash", candidates=1,
        clash_probe=_probe({0}, {7, 3}),
    )
    assert out[0] == 3


def test_a_rigid_move_can_be_applied_to_other_coordinates() -> None:
    """The transform is what the clash probe needs, not the superposed points."""
    rng = np.random.default_rng(0)
    pre = rng.normal(size=(12, 3))
    angle = 0.6
    rot = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    shift = np.array([1.5, -2.0, 0.25])
    post = pre @ rot.T + shift

    move = rigid_between(pre, post)
    # Recovered from one pair of point sets, then applied to a DIFFERENT set.
    other = rng.normal(size=(5, 3))
    assert np.allclose(apply_rigid(other, move), other @ rot.T + shift, atol=1e-8)
    assert np.allclose(apply_rigid(pre, move), post, atol=1e-8)

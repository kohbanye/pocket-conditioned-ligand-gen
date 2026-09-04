"""The ESM3 adapter must hand multi-chain targets over in ESM3's own format.

Feeding chains butt-joined, indexed by author residue number, is what produced
every apparent ESM3 reconstruction outlier on CASP16 (see
``docs/results/2026-09-04_esm3_chain_handling.md``). These lock the two
properties that fixed it, without needing the ESM3 weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from recon_bench.adapters.esm3 import chain_break_layout
from recon_bench.structio import Backbone


def _backbone(chain_ids: list[str], res_ids: list[int]) -> Backbone:
    n = len(chain_ids)
    coords = np.arange(n * 9, dtype=np.float64).reshape(n, 3, 3)
    return Backbone(
        coords=coords,
        seq="A" * n,
        res_ids=np.asarray(res_ids),
        chain_ids=np.asarray(chain_ids),
    )


def test_single_chain_gets_no_separator() -> None:
    bb = _backbone(["A"] * 4, [7, 8, 9, 10])
    coords, residue_index, is_residue, order = chain_break_layout(bb)

    assert is_residue.all()
    assert coords.shape == (4, 3, 3)
    np.testing.assert_array_equal(order, [0, 1, 2, 3])
    # ESM3's own default, not the author numbering the file happens to carry.
    np.testing.assert_array_equal(residue_index, [1, 2, 3, 4])


def test_chain_boundary_gets_one_separator_residue() -> None:
    bb = _backbone(["A", "A", "B", "B"], [1, 2, 1, 2])
    coords, residue_index, is_residue, order = chain_break_layout(bb)

    # One extra row, and it sits between the chains.
    assert len(coords) == 5
    np.testing.assert_array_equal(is_residue, [True, True, False, True, True])
    # The separator is what ESM3 masks out: NaN coordinates, residue_index -1.
    assert np.isnan(coords[2]).all()
    assert not np.isnan(coords[is_residue]).any()
    assert residue_index[2] == -1
    # Real residues stay on one strictly increasing run, so no two residues ever
    # reach the relative-position embedding at offset 0.
    real = residue_index[is_residue]
    assert (np.diff(real) > 0).all()
    assert len(set(real.tolist())) == len(real)
    # ``order`` maps every emitted residue back to its input row.
    np.testing.assert_array_equal(order, [0, 1, 2, 3])


def test_author_numbering_that_collides_across_chains_is_not_passed_through() -> None:
    # The CASP16 dimers: both chains numbered 1..3.
    bb = _backbone(["A", "A", "A", "B", "B", "B"], [1, 2, 3, 1, 2, 3])
    _, residue_index, is_residue, _ = chain_break_layout(bb)

    real = residue_index[is_residue]
    assert len(set(real.tolist())) == 6, "chain B must not reuse chain A's indices"


def test_interleaved_chains_are_grouped_and_order_tracks_it() -> None:
    bb = _backbone(["A", "B", "A", "B"], [1, 1, 2, 2])
    coords, residue_index, is_residue, order = chain_break_layout(bb)

    # Rows are regrouped A,A | sep | B,B -- and ``order`` says which input row
    # each one came from, so ref/res_keys stay aligned with the prediction.
    np.testing.assert_array_equal(order, [0, 2, 1, 3])
    np.testing.assert_array_equal(is_residue, [True, True, False, True, True])
    np.testing.assert_array_equal(coords[is_residue], bb.coords[order])


def test_three_chains_get_two_separators() -> None:
    bb = _backbone(["A", "B", "C"], [1, 1, 1])
    _, _, is_residue, order = chain_break_layout(bb)

    assert (~is_residue).sum() == 2
    assert len(order) == 3


@pytest.mark.parametrize("chains", [["A"], ["A", "B"], ["A", "B", "C"]])
def test_every_real_residue_survives(chains: list[str]) -> None:
    bb = _backbone(chains, list(range(1, len(chains) + 1)))
    _, _, is_residue, order = chain_break_layout(bb)

    assert is_residue.sum() == len(bb) == len(order)

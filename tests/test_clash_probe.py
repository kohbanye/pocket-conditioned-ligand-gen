"""The clash probe reads geometry only, and reads it the way the bench does."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from prolit.api import make_clash_probe, vdw_radii
from prolit.chem.rigid_fit import CLASH_FRACTION
from prolit.model.mlm_decode import refine_codes
from tests.test_mlm_decode import MASK_ID, NC, _probe, _Ranked

if TYPE_CHECKING:
    from collections.abc import Callable


def _decode_fixed(
    coords: np.ndarray, radius: float = 1.7
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """A decode that ignores the codes: positions are whatever we say."""

    def decode(seqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b, n = np.asarray(seqs).shape
        return (
            np.repeat(coords[None], b, axis=0),
            np.full((b, n), radius),
        )

    return decode


def test_an_atom_far_from_the_protein_is_clear() -> None:
    lig = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    probe = make_clash_probe(_decode_fixed(lig), np.array([[100.0, 0, 0]]), ["C"])
    assert not probe(np.zeros((1, 2), dtype=int)).any()


def test_an_atom_inside_the_wall_is_flagged() -> None:
    lig = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    probe = make_clash_probe(_decode_fixed(lig), np.array([[0.0, 0, 0]]), ["C"])
    out = probe(np.zeros((1, 2), dtype=int))
    assert out[0, 0]


def test_the_threshold_is_the_deployed_one() -> None:
    """Just outside CLASH_FRACTION of the summed radii is not a clash."""
    r_lig, r_rec = 1.7, float(vdw_radii(["C"])[0])
    limit = CLASH_FRACTION * (r_lig + r_rec)
    lig = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    for factor, expected in ((0.99, True), (1.01, False)):
        probe = make_clash_probe(
            _decode_fixed(lig, r_lig), np.array([[limit * factor, 0, 0]]), ["C"]
        )
        assert bool(probe(np.zeros((1, 2), dtype=int))[0, 0]) is expected


def test_receptor_hydrogens_are_ignored() -> None:
    """The criterion is calibrated on heavy atoms; an H must not trigger it."""
    lig = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    probe = make_clash_probe(
        _decode_fixed(lig), np.array([[0.0, 0, 0], [50.0, 0, 0]]), ["H", "C"]
    )
    assert not probe(np.zeros((1, 2), dtype=int)).any()


def test_every_row_of_the_batch_is_answered_independently() -> None:
    def decode(seqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Row k puts its first atom k Angstrom away from the receptor atom.
        b, n = np.asarray(seqs).shape
        xyz = np.zeros((b, n, 3))
        for k in range(b):
            xyz[k, 0] = [float(k) * 4.0, 0.0, 0.0]
            xyz[k, 1] = [50.0, 0.0, 0.0]
        return xyz, np.full((b, n), 1.7)

    probe = make_clash_probe(decode, np.array([[0.0, 0, 0]]), ["C"])
    out = probe(np.zeros((4, 2), dtype=int))
    assert out[0, 0]
    assert not out[1:, 0].any()
    assert not out[:, 1].any()


def test_bonds_are_only_judged_when_asked_for() -> None:
    """Selection asks "in the wall"; replacement asks "and still a molecule"."""
    lig = np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]])  # a 9 A "bond"
    far = np.array([[100.0, 0, 0]])
    assert not make_clash_probe(_decode_fixed(lig), far, ["C"])(
        np.zeros((1, 2), dtype=int)
    ).any()
    with_bonds = make_clash_probe(_decode_fixed(lig), far, ["C"], bonds=[(0, 1)])
    assert with_bonds(np.zeros((1, 2), dtype=int)).all()


def test_a_bond_inside_the_window_is_accepted() -> None:
    lig = np.array([[0.0, 0.0, 0.0], [1.45, 0.0, 0.0]])
    probe = make_clash_probe(
        _decode_fixed(lig), np.array([[100.0, 0, 0]]), ["C"], bonds=[(0, 1)]
    )
    assert not probe(np.zeros((1, 2), dtype=int)).any()


def test_a_too_short_bond_is_rejected() -> None:
    lig = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
    probe = make_clash_probe(
        _decode_fixed(lig), np.array([[100.0, 0, 0]]), ["C"], bonds=[(0, 1)]
    )
    assert probe(np.zeros((1, 2), dtype=int)).all()


def test_the_picker_uses_the_accept_probe_not_the_clash_probe() -> None:
    """A candidate that clears the wall but breaks a bond must not be picked."""
    out = refine_codes(
        _Ranked([3, 4, 5]), MASK_ID, [1, 2], [7] * 10, codebook_size=NC,
        rounds=1, frac=1.0, order="clash",
        clash_probe=_probe({0}, {7}),          # position 0 is in the wall
        accept_probe=_probe({0}, {7, 3}),      # 3 clears it but breaks a bond
    )
    assert out[0] == 4



def test_a_narrow_angle_is_rejected() -> None:
    """Keeping the bond is not enough: the replacement must not fold the angle."""
    # 0 bonded to 1 and 2; placing them 40 degrees apart is far below the window.
    r = 1.45
    lig = np.array([
        [0.0, 0.0, 0.0],
        [r, 0.0, 0.0],
        [r * math.cos(math.radians(40)), r * math.sin(math.radians(40)), 0.0],
    ])
    probe = make_clash_probe(
        _decode_fixed(lig), np.array([[100.0, 0, 0]]), ["C"], bonds=[(0, 1), (0, 2)]
    )
    assert probe(np.zeros((1, 3), dtype=int))[0, 0]


def test_a_normal_angle_is_accepted() -> None:
    r = 1.45
    lig = np.array([
        [0.0, 0.0, 0.0],
        [r, 0.0, 0.0],
        [r * math.cos(math.radians(109.5)), r * math.sin(math.radians(109.5)), 0.0],
    ])
    probe = make_clash_probe(
        _decode_fixed(lig), np.array([[100.0, 0, 0]]), ["C"], bonds=[(0, 1), (0, 2)]
    )
    assert not probe(np.zeros((1, 3), dtype=int)).any()


def test_a_non_bonded_atom_too_close_is_rejected() -> None:
    """1-4 and beyond must keep their distance; 1-2 and 1-3 are exempt."""
    lig = np.array([
        [0.0, 0.0, 0.0],   # 0
        [1.45, 0.0, 0.0],  # 1, bonded to 0
        [2.9, 0.0, 0.0],   # 2, bonded to 1  (1-3 from 0, exempt)
        [1.9, 0.0, 0.0],   # 3, bonded to nothing near 0 -> 1.9 A is too close
    ])
    probe = make_clash_probe(
        _decode_fixed(lig), np.array([[100.0, 0, 0]]), ["C"], bonds=[(0, 1), (1, 2)]
    )
    assert probe(np.zeros((1, 4), dtype=int))[0, 0]


def test_only_restricts_the_answer_to_one_column() -> None:
    """The picker reads one column; computing the rest is wasted decode."""
    lig = np.array([[0.0, 0.0, 0.0], [1.45, 0.0, 0.0]])
    probe = make_clash_probe(_decode_fixed(lig), np.array([[0.0, 0, 0]]), ["C"])
    full = probe(np.zeros((1, 2), dtype=int))
    one = probe(np.zeros((1, 2), dtype=int), 0)
    assert full[0, 0]
    assert one[0, 0]
    assert not one[0, 1]

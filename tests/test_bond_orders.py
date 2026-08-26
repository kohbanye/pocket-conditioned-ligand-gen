"""Bond orders recovered from valence, without a bond-order token.

The claim these tests pin down is that ``element + charge + numH`` plus a
heavy-atom graph determines the bond orders, so the tokenizer never has to
spend bits on them. Each case below is one way that claim could be false.
"""

from __future__ import annotations

import numpy as np
import pytest

from prolit.api import assign_bond_orders, mol_from_decoded, target_bond_sums

RING6 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]


def test_benzene_kekulizes() -> None:
    """Six CH in a ring must alternate; an all-single ring cannot satisfy C."""
    order = assign_bond_orders(["C"] * 6, [0] * 6, [1] * 6, RING6)
    assert order is not None
    assert sorted(order.values()) == [1, 1, 1, 2, 2, 2]


def test_cyclohexane_stays_single() -> None:
    """The same ring with CH2 has no excess valence, so nothing is promoted."""
    order = assign_bond_orders(["C"] * 6, [0] * 6, [2] * 6, RING6)
    assert order is not None
    assert set(order.values()) == {1}


def test_nitrile_is_triple() -> None:
    """C#N needs one bond to carry two promotions, not two bonds one each."""
    order = assign_bond_orders(["C", "C", "N"], [0, 0, 0], [3, 0, 0], [(0, 1), (1, 2)])
    assert order == {(0, 1): 1, (1, 2): 3}


def test_charge_shifts_the_valence() -> None:
    """A charged N takes four bonds where a neutral one takes three."""
    assert target_bond_sums("N", 0, 0) == (3,)
    assert target_bond_sums("N", 1, 0) == (4,)
    assert target_bond_sums("O", -1, 0) == (1,)


def test_sulfonamide_picks_the_hypervalent_sulfur() -> None:
    """S is 2-, 4- or 6-valent; only 6 closes here, so the search must reach it."""
    # O=S(=O)(C)N : S bonded to two O, one C, one N.
    order = assign_bond_orders(
        ["S", "O", "O", "C", "N"],
        [0] * 5,
        [0, 0, 0, 3, 2],
        [(0, 1), (0, 2), (0, 3), (0, 4)],
    )
    assert order is not None
    assert sorted(order.values()) == [1, 1, 2, 2]


def test_impossible_chemistry_returns_none() -> None:
    """Contradiction is reported, not papered over with single bonds.

    A carbon with four heavy neighbours AND two hydrogens is over-valent. The
    honest answer is that the decoded chemistry and the graph disagree.
    """
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4)]
    assert (
        assign_bond_orders(["C"] * 5, [0] * 5, [2, 3, 3, 3, 3], bonds) is None
    )


def test_unknown_element_returns_none() -> None:
    assert target_bond_sums("Xx", 0, 0) == ()
    assert assign_bond_orders(["Xx", "C"], [0, 0], [0, 3], [(0, 1)]) is None


def test_mol_from_decoded_gives_aromatic_benzene() -> None:
    """RDKit's own perception labels the ring; the solver only kekulizes it."""
    chem = pytest.importorskip("rdkit.Chem")
    angles = np.arange(6) * np.pi / 3
    coords = np.stack(
        [1.39 * np.cos(angles), 1.39 * np.sin(angles), np.zeros(6)], axis=1
    )
    mol = mol_from_decoded(["C"] * 6, [0] * 6, [1] * 6, coords)
    assert mol is not None
    assert chem.MolToSmiles(mol) == "c1ccccc1"
    assert all(a.GetIsAromatic() for a in mol.GetAtoms())


def test_mol_from_decoded_perceives_bonds_when_not_given() -> None:
    """Without a graph it falls back to distance perception over the coords."""
    coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    mol = mol_from_decoded(["C", "O"], [0, 0], [2, 0], coords)
    assert mol is not None
    assert mol.GetNumBonds() == 1
    assert mol.GetBondWithIdx(0).GetBondTypeAsDouble() == 2.0

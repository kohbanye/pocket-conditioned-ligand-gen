"""Pruning must remove the bond perception was least sure of, and only that.

The point of the pruning is to keep a molecule inside the chemistry-aware path
when distance perception over-coordinates one atom. If it prunes a graph that
was already fine, or strands an atom, it trades one PoseBusters failure for
another.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from prolit.chem.bond_orders import (
    connect_fragments,
    mol_from_decoded,
    prune_to_valence,
)


def _line(n: int, spacing: float = 1.5) -> np.ndarray:
    return np.stack([np.arange(n) * spacing, np.zeros(n), np.zeros(n)], axis=1)


def test_a_valid_graph_is_untouched() -> None:
    elements = ["C", "C", "C"]
    coords = _line(3)
    bonds = [(0, 1), (1, 2)]
    assert prune_to_valence(elements, [0, 0, 0], [3, 2, 3], bonds, coords) == bonds


def test_the_longest_relative_bond_goes_first() -> None:
    """Among cuts that keep the molecule whole, the doubtful bond is the one."""
    # A six-ring, plus a chord across it that perception invented. Nothing in
    # a ring is a bridge, so every bond is a legal cut and the choice is made
    # on length alone -- and the chord is much the longest.
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    coords = np.stack(
        [1.45 * np.cos(angles), 1.45 * np.sin(angles), np.zeros(6)], axis=1
    )
    elements = ["C"] * 6
    ring = [(i, (i + 1) % 6) for i in range(6)]
    chord = (0, 3)
    bonds = [*ring, chord]
    kept = prune_to_valence(elements, [0] * 6, [2] * 6, bonds, coords)
    assert chord not in kept
    assert len(kept) == len(bonds) - 1


def test_a_bridge_is_never_cut() -> None:
    """Two rings joined by a single bond: the joining bond stays put."""
    def ring(centre_x: float) -> np.ndarray:
        a = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
        return np.stack(
            [centre_x + 1.45 * np.cos(a), 1.45 * np.sin(a), np.zeros(6)], axis=1
        )

    coords = np.vstack([ring(0.0), ring(4.4)])
    elements = ["C"] * 12
    bonds = [(i, (i + 1) % 6) for i in range(6)]
    bonds += [(6 + i, 6 + (i + 1) % 6) for i in range(6)]
    bridge = (0, 6)                       # the only path between the rings
    bonds.append(bridge)
    kept = prune_to_valence(elements, [0] * 12, [2] * 12, bonds, coords)
    assert bridge in kept


def test_every_atom_ends_within_its_valence() -> None:
    rng = np.random.default_rng(0)
    coords = rng.normal(0.0, 1.1, size=(12, 3))
    elements = ["C"] * 8 + ["N"] * 2 + ["O"] * 2
    bonds = [(i, j) for i in range(12) for j in range(i + 1, 12)
             if np.linalg.norm(coords[i] - coords[j]) < 1.8]
    num_h = [0] * 12
    kept = prune_to_valence(elements, [0] * 12, num_h, bonds, coords)
    limit = {"C": 4, "N": 3, "O": 2}
    degree: dict[int, int] = {}
    for i, j in kept:
        degree[i] = degree.get(i, 0) + 1
        degree[j] = degree.get(j, 0) + 1
    for i, element in enumerate(elements):
        assert degree.get(i, 0) <= limit[element]


def test_pruning_rescues_a_molecule_the_solver_would_refuse() -> None:
    """The mechanism, end to end: over-coordinated -> None, pruned -> a Mol.

    The realistic shape: a ring whose two ends the decoder placed close enough
    that perception bridges them, giving a ring carbon a fifth neighbour. The
    spurious bond is the longest one there, and cutting it strands nothing.
    """
    # Hexane, with the chain bent so atoms 0 and 5 nearly touch.
    angles = np.linspace(0.0, np.pi * 1.15, 6)
    coords = np.stack(
        [1.45 * np.cos(angles), 1.45 * np.sin(angles), np.zeros(6)], axis=1
    )
    elements = ["C"] * 6
    charges = [0] * 6
    num_h = [3, 2, 2, 2, 2, 3]
    chain = [(i, i + 1) for i in range(5)]
    bridged = [*chain, (0, 5)]      # perception's extra bond closes the ring
    # As a ring every carbon needs one more bond than its numH allows.
    assert mol_from_decoded(elements, charges, num_h, coords, bridged) is None
    kept = prune_to_valence(elements, charges, num_h, bridged, coords)
    assert kept == chain
    assert mol_from_decoded(elements, charges, num_h, coords, kept) is not None


def test_a_forced_stranding_gives_up_rather_than_cutting() -> None:
    coords = np.array([
        [0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [-1.5, 0.0, 0.0],
        [0.0, 1.5, 0.0], [0.0, -1.5, 0.0], [0.0, 0.0, 1.55],
    ])
    elements = ["C"] * 6
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
    # Every bond is some outer atom's only bond, so there is no cut that does
    # not strand one. The graph comes back as it went in.
    kept = prune_to_valence(elements, [0] * 6, [0, 3, 3, 3, 3, 3], bonds, coords)
    assert kept == bonds


def test_a_whole_molecule_is_not_given_extra_bonds() -> None:
    elements = ["C", "C", "C"]
    coords = _line(3)
    bonds = [(0, 1), (1, 2)]
    assert connect_fragments(elements, bonds, coords) == bonds


def test_two_pieces_are_joined_at_their_shortest_gap() -> None:
    # Two ethane-ish fragments; the gap between atoms 1 and 2 is the shortest
    # link between the pieces, so that is the one that closes.
    coords = np.array([
        [0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
        [3.4, 0.0, 0.0], [4.9, 0.0, 0.0],
    ])
    elements = ["C"] * 4
    bonds = [(0, 1), (2, 3)]
    joined = connect_fragments(elements, bonds, coords)
    assert (1, 2) in joined
    assert len(joined) == len(bonds) + 1


def test_joining_then_pruning_leaves_one_piece_within_valence() -> None:
    rng = np.random.default_rng(5)
    coords = rng.normal(0.0, 1.3, size=(10, 3))
    elements = ["C"] * 10
    bonds = [(i, j) for i in range(10) for j in range(i + 1, 10)
             if np.linalg.norm(coords[i] - coords[j]) < 1.6]
    joined = connect_fragments(elements, bonds, coords)
    kept = prune_to_valence(elements, [0] * 10, [0] * 10, joined, coords)

    parent = list(range(10))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in kept:
        parent[find(i)] = find(j)
    assert len({find(i) for i in range(10)}) == 1

    degree: dict[int, int] = {}
    for i, j in kept:
        degree[i] = degree.get(i, 0) + 1
        degree[j] = degree.get(j, 0) + 1
    assert all(degree.get(i, 0) <= 4 for i in range(10))


def _benzene_coords() -> np.ndarray:
    a = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    return np.stack([1.39 * np.cos(a), 1.39 * np.sin(a), np.zeros(6)], axis=1)


def test_the_models_aromatic_call_overrules_a_wrong_numh() -> None:
    """numH one too high forces an aliphatic ring; the aromatic head wins."""
    coords = _benzene_coords()
    elements = ["C"] * 6
    bonds = [(i, (i + 1) % 6) for i in range(6)]
    wrong = [2] * 6                      # the decoder's numH said CH2

    # Without the head, numH is taken at face value and the ring saturates.
    plain = mol_from_decoded(elements, [0] * 6, wrong, coords, bonds, perceived=True)
    assert plain is not None
    assert not any(a.GetIsAromatic() for a in plain.GetAtoms())

    fixed = mol_from_decoded(
        elements, [0] * 6, wrong, coords, bonds,
        perceived=True, aromatic=[True] * 6,
    )
    assert fixed is not None
    assert Chem.MolToSmiles(fixed) == "c1ccccc1"


def test_atoms_the_model_calls_aliphatic_are_left_alone() -> None:
    coords = _benzene_coords()
    mol = mol_from_decoded(
        ["C"] * 6, [0] * 6, [2] * 6, coords, perceived=True, aromatic=[False] * 6
    )
    assert mol is not None
    assert not any(a.GetIsAromatic() for a in mol.GetAtoms())


def test_a_substituted_aromatic_carbon_gives_back_two_hydrogens() -> None:
    """Toluene: the ring carbon bearing the methyl is forced twice over."""
    ring = _benzene_coords()
    methyl = ring[0] * (1.0 + 1.50 / 1.39)
    coords = np.vstack([ring, methyl[None, :]])
    wrong = [2, 2, 2, 2, 2, 2, 3]
    mol = mol_from_decoded(
        ["C"] * 7, [0] * 7, wrong, coords,
        perceived=True, aromatic=[True] * 6 + [False],
    )
    assert mol is not None
    assert Chem.MolToSmiles(mol) == "Cc1ccccc1"


def test_a_known_graph_keeps_its_own_chemistry() -> None:
    """Reconstruction passes real bonds; the head must not rewrite numH."""
    coords = _benzene_coords()
    bonds = [(i, (i + 1) % 6) for i in range(6)]
    mol = mol_from_decoded(
        ["C"] * 6, [0] * 6, [2] * 6, coords, bonds, aromatic=[True] * 6
    )
    assert mol is not None
    assert not any(a.GetIsAromatic() for a in mol.GetAtoms())


def test_correct_numh_is_not_disturbed() -> None:
    coords = _benzene_coords()
    mol = mol_from_decoded(
        ["C"] * 6, [0] * 6, [1] * 6, coords, perceived=True, aromatic=[True] * 6
    )
    assert mol is not None
    assert Chem.MolToSmiles(mol) == "c1ccccc1"

"""The generation order must be a property of the molecule, not of the file.

Until ``buried_first`` existed the language model emitted atoms in whatever
order the SDF stored them, so "the next atom" carried no spatial meaning. The
cost was measured on generated ligands: at the *same* distance from the ligand
centroid, atoms late in the sequence clash 1.35-2.05x as often as early ones.
"""

from __future__ import annotations

import numpy as np

from prolit.tokenizers.atom import _buried_first_order


def _chain(n: int = 6, spacing: float = 1.5) -> list[tuple[str, float, float, float]]:
    return [("C", i * spacing, 0.0, 0.0) for i in range(n)]


def _bonds(n: int) -> list[tuple[int, int, int]]:
    return [(i, i + 1, 1) for i in range(n - 1)]


def test_it_starts_at_the_atom_nearest_the_pocket() -> None:
    atoms = _chain(6)
    centroid = np.array([0.0, 0.0, 0.0])          # nearest atom 0
    order = _buried_first_order(atoms, _bonds(6), list(range(6)), centroid)
    assert order[0] == 0
    assert order == list(range(6))

    centroid = np.array([7.5, 0.0, 0.0])          # nearest atom 5
    order = _buried_first_order(atoms, _bonds(6), list(range(6)), centroid)
    assert order[0] == 5
    assert order == list(reversed(range(6)))


def test_the_order_does_not_depend_on_the_file_order() -> None:
    """Shuffling the input must give the same sequence of atoms."""
    atoms = _chain(6)
    centroid = np.array([0.0, 0.0, 0.0])
    straight = _buried_first_order(atoms, _bonds(6), list(range(6)), centroid)

    perm = [3, 0, 5, 1, 4, 2]
    shuffled = [atoms[i] for i in perm]
    fwd = {old: new for new, old in enumerate(perm)}
    bonds = [(fwd[a], fwd[b], t) for a, b, t in _bonds(6)]
    got = _buried_first_order(shuffled, bonds, list(range(6)), centroid)

    # Compare the atoms themselves, not the indices into two different lists.
    assert [atoms[i] for i in straight] == [shuffled[i] for i in got]


def test_every_atom_after_the_first_is_bonded_to_one_already_placed() -> None:
    rng = np.random.default_rng(0)
    n = 12
    coords = np.cumsum(rng.normal(0.0, 1.4, size=(n, 3)), axis=0)
    atoms = [("C", *c) for c in coords]
    bonds = [(i, i + 1, 1) for i in range(n - 1)]
    bonds.extend([(0, 5, 1), (3, 9, 1)])          # a couple of rings
    order = _buried_first_order(atoms, bonds, list(range(n)), coords.mean(0))

    adjacency: dict[int, set[int]] = {i: set() for i in range(n)}
    for a, b, _ in bonds:
        adjacency[a].add(b)
        adjacency[b].add(a)
    placed = {order[0]}
    for atom in order[1:]:
        assert adjacency[atom] & placed, f"{atom} starts a new island"
        placed.add(atom)


def test_disconnected_pieces_are_all_emitted() -> None:
    atoms = [*_chain(3), ("C", 20.0, 0.0, 0.0), ("C", 21.5, 0.0, 0.0)]
    bonds = [(0, 1, 1), (1, 2, 1), (3, 4, 1)]
    order = _buried_first_order(atoms, bonds, list(range(5)), np.zeros(3))
    assert sorted(order) == list(range(5))
    assert order[:3] == [0, 1, 2]                 # the near fragment first

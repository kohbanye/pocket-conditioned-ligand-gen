"""Turn the ligand's rotatable bonds so it settles into the pocket.

:mod:`prolit.chem.rigid_fit` moves the whole molecule and stops there, and the
measurement says that is not enough: after the rigid placement, the generated
ligands score -1.19 by Vina where a local optimisation of the same pose reaches
-4.31. Three kilocalories sit between them, and Vina's local optimisation
differs from the rigid one in exactly one way -- it turns the torsions too.

So this adds them. A dihedral rotation about a single acyclic bond leaves every
bond length, every bond angle, and every ring untouched, so the same guarantee
the rigid step has carries over: **PoseBusters' bond-length, bond-angle and
ring checks are invariant by construction**, and only the steric ones can move.
That is what separates this from the flexible pocket-aware relaxation
:mod:`prolit.chem.relax` records as a failure, which relieved clashes by
bending the molecule and paid 0.932 -> 0.851 in PoseBusters geometry.

The objective is the one :mod:`prolit.chem.rigid_fit` already defines, extended
inward: the same Lennard-Jones form now also runs over ligand atom pairs
separated by more than three bonds, which is what stops a torsion from folding
the molecule onto itself. Same sigma from the same radii, same unit epsilon --
turning a bond is free covalently, so nothing here needs a weight to say how
much a turn should cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from prolit.chem.rigid_fit import (
    _INFEASIBLE,
    _NEGLIGIBLE,
    _clamp,
    _Overlap,
    _rotation,
    _search,
    vdw_radii,
)

if TYPE_CHECKING:
    from rdkit.Chem import Mol

#: Pairs closer than this along the bond graph are held rigid by the bond
#: angles, so their separation is not something a torsion can change.
_MIN_TOPOLOGICAL_SEPARATION = 4


def rotatable_bonds(mol: Mol) -> list[tuple[int, int, np.ndarray]]:
    """Acyclic single bonds with something to turn on both sides.

    Returns ``(begin, end, moving)`` where ``moving`` indexes the atoms that
    follow ``end`` -- the smaller side, so a turn moves as few atoms as
    possible and the pose stays recognisable.
    """
    from rdkit import Chem  # noqa: PLC0415

    out: list[tuple[int, int, np.ndarray]] = []
    n = mol.GetNumAtoms()
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if mol.GetAtomWithIdx(begin).GetDegree() < 2:  # noqa: PLR2004
            continue
        if mol.GetAtomWithIdx(end).GetDegree() < 2:  # noqa: PLR2004
            continue
        side = _side_of(mol, begin, end, n)
        if side is None:
            continue
        other = np.setdiff1d(np.arange(n), side)
        smaller = side if len(side) <= len(other) else other
        # Whichever side is smaller, the axis runs the same way; name the
        # pivot so the rotation is about the bond and not about the origin.
        pivot, tip = (begin, end) if smaller is side else (end, begin)
        out.append((pivot, tip, smaller[smaller != pivot]))
    return out


def _side_of(mol: Mol, begin: int, end: int, n: int) -> np.ndarray | None:
    """Atoms reachable from ``end`` without crossing ``begin``-``end``."""
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if {i, j} == {begin, end}:
            continue
        adjacency[i].append(j)
        adjacency[j].append(i)
    seen = {end}
    stack = [end]
    while stack:
        node = stack.pop()
        for neighbour in adjacency[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    if begin in seen:
        return None  # the bond was in a ring after all
    return np.fromiter(sorted(seen), dtype=int)


def far_pairs(mol: Mol) -> tuple[np.ndarray, np.ndarray]:
    """Ligand atom index pairs more than three bonds apart, and their sigma."""
    from rdkit import Chem  # noqa: PLC0415

    distances = Chem.GetDistanceMatrix(mol)
    radii = vdw_radii([a.GetSymbol() for a in mol.GetAtoms()])
    i, j = np.triu_indices(mol.GetNumAtoms(), k=1)
    keep = distances[i, j] >= _MIN_TOPOLOGICAL_SEPARATION
    pairs = np.stack([i[keep], j[keep]], axis=1)
    if len(pairs) == 0:
        return pairs.reshape(0, 2), np.zeros(0)
    return pairs, radii[pairs[:, 0]] + radii[pairs[:, 1]]


def _apply(
    coords: np.ndarray,
    torsions: list[tuple[int, int, np.ndarray]],
    angles: np.ndarray,
) -> np.ndarray:
    out = coords.copy()
    for (pivot, tip, moving), angle in zip(torsions, angles, strict=True):
        if abs(angle) < _NEGLIGIBLE or len(moving) == 0:
            continue
        axis = out[tip] - out[pivot]
        norm = float(np.linalg.norm(axis))
        if norm < _NEGLIGIBLE:
            continue
        matrix = _rotation(axis / norm * angle)
        out[moving] = (out[moving] - out[pivot]) @ matrix.T + out[pivot]
    return out


def _self_energy(coords: np.ndarray, pairs: np.ndarray, sigma: np.ndarray) -> float:
    """Lennard-Jones over the ligand's own far-apart pairs."""
    if len(pairs) == 0:
        return 0.0
    distance = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
    ratio = sigma / np.maximum(distance, 0.4)
    six = ratio**6
    return float(np.sum(six * six - 2.0 * six))


def torsion_pocket_fit(  # noqa: PLR0913
    mol: Mol,
    receptor_coords: np.ndarray,
    receptor_radii: np.ndarray,
    *,
    max_translation: float = 2.5,
    max_rotation_deg: float = 30.0,
    n_restarts: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """Return coordinates after a rigid placement plus a torsional settle.

    The molecule's covalent geometry is untouched: only the six rigid degrees
    of freedom and the acyclic single-bond dihedrals move.
    """
    coords = mol.GetConformer().GetPositions()
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    heavy = np.array([e != "H" for e in elements])
    if not heavy.any() or len(receptor_coords) == 0:
        return coords

    torsions = rotatable_bonds(mol)
    pairs, sigma = far_pairs(mol)
    ligand_radii = vdw_radii(elements)
    overlap = _Overlap(receptor_coords, receptor_radii, ligand_radii[heavy])
    well = _Overlap(receptor_coords, receptor_radii, ligand_radii[heavy], "lj")

    centre = coords.mean(axis=0)
    max_rotation = np.deg2rad(max_rotation_deg)
    n_torsions = len(torsions)

    def place(params: np.ndarray) -> np.ndarray:
        turned = _apply(coords, torsions, params[6:]) if n_torsions else coords
        translation, rotation = _clamp(params[:6], max_translation, max_rotation)
        return (turned - centre) @ _rotation(rotation).T + centre + translation

    def relieve(params: np.ndarray) -> float:
        moved = place(params)
        return overlap(moved[heavy]) + max(_self_energy(moved, pairs, sigma), 0.0)

    start = np.zeros(6 + n_torsions)
    best = _search(relieve, n_restarts, seed, start=start)
    ceiling = relieve(best) + _NEGLIGIBLE

    def snug(params: np.ndarray) -> float:
        moved = place(params)
        excess = relieve(params) - ceiling
        if excess > 0.0:
            return _INFEASIBLE + excess
        return well(moved[heavy]) + _self_energy(moved, pairs, sigma)

    return place(_search(snug, n_restarts, seed, start=best))

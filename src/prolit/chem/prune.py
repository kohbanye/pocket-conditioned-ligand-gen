"""Delete terminal ligand atoms that sit inside the pocket wall.

The gap to the reference ligand is 84% Vina ``repulsion`` (2026-09-10), and the
refiner has already taken what rigid motion can reach. What is left are atoms
the molecule cannot place anywhere: a generated ligand carries about one
terminal atom pushed through the protein surface, and the reference ligand
carries none.

Deleting a degree-one heavy atom is a closed operation -- the neighbour gains an
implicit hydrogen, the molecule stays connected, and nothing about the remaining
geometry changes. Deleting anything else is not, so nothing else is deleted.

Two rules that were measured and rejected, recorded so they are not retried:

* **Rank by Vina's own ``repulsion`` term.** Buys marginally more (paired 1.90 vs
  1.72 kcal of intermolecular energy) and is circular -- it selects atoms with
  the function the result is scored by, the same defect that disqualified Vina
  terms as a refiner teacher.
* **Spare nitrogen and oxygen.** Pruning deletes oxygen at **8.2x** the rate it
  deletes carbon
  (5.34% of all O against 0.65% of all C, one-to-one over 5858 molecules), and
  Vina's ``hbond`` term fires only on N/O -- so preferring to delete a carbon
  looks like free hydrogen bonds. It is not: requiring a polar leaf to beat the
  best non-polar one by a margin of 0.1, 0.5 and even 2.0 squared Angstroms
  leaves the deletions **identical, atom for atom**. The clashing-leaf set has no
  non-polar alternative to choose instead; the terminal atoms in the wall simply
  *are* oxygens (carbonyls and hydroxyls are what a degree-one filter leaves).
  There is no choice to make, so the rule cannot be made element-aware, and the
  parameter that showed this is not kept.
* **Delete a fixed number of atoms.** "The best two" buys 1.66 kcal but takes the
  molecule to 21.0 heavy atoms against the reference's 22, i.e. it buys score
  with size. Three buys 1.84 at 20.0. The rule below has no such knob: it stops
  when the molecule has no clashing terminal atom left, which is a property of
  the molecule, and lands at 21.8.

The control that makes this a defect of the model rather than a preference of
the scoring function: the same rule fires on 8% of reference ligands and makes
them slightly *worse*, while it fires on 50% of generated molecules and improves
them by 1.20 kcal (p=4e-14, better on 75% of targets).
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from scipy.spatial import KDTree

from prolit.chem.rigid_fit import CLASH_FRACTION, vdw_radii

__all__ = ["prune_clashing_leaves"]

#: A molecule is never pruned below this many heavy atoms, whatever it clashes
#: with. Nothing in the measurement hit this floor; it stops a pathological
#: input from being pruned to nothing.
MIN_HEAVY: int = 6



def _heavy_graph(mol: Chem.Mol) -> tuple[list[int], dict[int, set[int]]]:
    idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
    live = set(idx)
    adj: dict[int, set[int]] = {i: set() for i in idx}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in live and j in live:
            adj[i].add(j)
            adj[j].add(i)
    return idx, adj


def _overlap_and_clash(
    coords: np.ndarray,
    radii: np.ndarray,
    tree: KDTree,
    rec_xyz: np.ndarray,
    rec_radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Summed squared van der Waals overlap, and whether the atom clashes."""
    overlap = np.zeros(len(coords))
    clash = np.zeros(len(coords), dtype=bool)
    reach = float(rec_radii.max())
    for i, (c, r) in enumerate(zip(coords, radii, strict=True)):
        for j in tree.query_ball_point(c, r + reach):
            summed = r + rec_radii[j]
            d = float(np.linalg.norm(c - rec_xyz[j]))
            if d < summed:
                overlap[i] += (summed - d) ** 2
            if d < CLASH_FRACTION * summed:
                clash[i] = True
    return overlap, clash


def prune_clashing_leaves(
    mol: Chem.Mol,
    receptor_coords: np.ndarray,
    receptor_elements: list[str],
    *,
    max_deletions: int = 8,
) -> tuple[Chem.Mol, int]:
    """``mol`` with its clashing terminal heavy atoms removed.

    Repeatedly deletes the degree-one heavy atom that clashes with the receptor
    and has the largest van der Waals overlap, until no terminal atom clashes.
    Returns the pruned molecule and how many atoms were removed; the molecule is
    returned unchanged (and the count is zero) when nothing clashes, which is the
    common case for a well-placed ligand.

    ``max_deletions`` is a guard, not a tuning knob -- the rule stops on its own.
    """
    if mol.GetNumConformers() == 0:
        return mol, 0
    rec_xyz = np.asarray(receptor_coords, dtype=float)
    keep = np.array([e != "H" for e in receptor_elements])
    if keep.any():
        rec_xyz, receptor_elements = rec_xyz[keep], [
            e for e, k in zip(receptor_elements, keep, strict=True) if k
        ]
    rec_radii = vdw_radii(list(receptor_elements))
    tree = KDTree(rec_xyz)

    work = Chem.RWMol(mol)
    removed = 0
    for _ in range(max_deletions):
        idx, adj = _heavy_graph(work)
        if len(idx) <= MIN_HEAVY:
            break
        conf = work.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(i)) for i in idx])
        radii = vdw_radii([work.GetAtomWithIdx(i).GetSymbol() for i in idx])
        overlap, clash = _overlap_and_clash(coords, radii, tree, rec_xyz, rec_radii)
        leaves = [n for n, i in enumerate(idx) if len(adj[i]) == 1 and clash[n]]
        if not leaves:
            break
        worst = max(leaves, key=lambda n: overlap[n])
        work.RemoveAtom(idx[worst])
        removed += 1
    if removed == 0:
        return mol, 0
    out = work.GetMol()
    try:
        Chem.SanitizeMol(out)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError):
        return mol, 0
    if len(Chem.GetMolFrags(out)) != 1:
        return mol, 0
    return out, removed

"""The clashing-leaf pruner deletes what it should and nothing else."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from prolit.api import prune_clashing_leaves
from prolit.chem.rigid_fit import CLASH_FRACTION, vdw_radii


def _mol(smiles: str = "CCCCCCCCO") -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    return Chem.RemoveHs(mol)


def _receptor_on(
    atom_xyz: np.ndarray, element: str = "C"
) -> tuple[np.ndarray, list[str]]:
    """One receptor carbon placed close enough to ``atom_xyz`` to count as a clash."""
    return np.array([atom_xyz]), [element]


def test_nothing_clashing_is_left_alone() -> None:
    mol = _mol()
    far = np.array([[100.0, 100.0, 100.0]])
    out, n = prune_clashing_leaves(mol, far, ["C"])
    assert n == 0
    assert out.GetNumAtoms() == mol.GetNumAtoms()
    assert out is mol


def test_a_clashing_terminal_atom_is_removed() -> None:
    mol = _mol()
    conf = mol.GetConformer()
    idx = [a.GetIdx() for a in mol.GetAtoms()]
    leaf = next(i for i in idx if mol.GetAtomWithIdx(i).GetDegree() == 1)
    rec, els = _receptor_on(np.array(list(conf.GetAtomPosition(leaf))))
    out, n = prune_clashing_leaves(mol, rec, els)
    assert n >= 1
    assert out.GetNumAtoms() == mol.GetNumAtoms() - n


def test_a_clashing_ring_atom_is_kept() -> None:
    """A ring atom is never a leaf, so it survives however badly it clashes.

    A chain is the wrong shape to assert this with: deleting a terminal atom
    turns its neighbour into a terminal atom, so an iterative rule eats inward
    along a chain by design (``MIN_HEAVY`` is what stops it). A ring atom has
    degree two no matter what is deleted around it.
    """
    mol = _mol("c1ccccc1CCO")
    conf = mol.GetConformer()
    ring = next(a.GetIdx() for a in mol.GetAtoms() if a.IsInRing())
    at = np.array(list(conf.GetAtomPosition(ring)))
    out, _ = prune_clashing_leaves(mol, *_receptor_on(at))
    oc = out.GetConformer()
    kept = [np.array(list(oc.GetAtomPosition(i))) for i in range(out.GetNumAtoms())]
    assert min(float(np.linalg.norm(k - at)) for k in kept) < 1e-6
    assert sum(1 for a in out.GetAtoms() if a.IsInRing()) == 6


def test_the_result_stays_connected_and_sanitizable() -> None:
    mol = _mol("c1ccccc1CCO")
    conf = mol.GetConformer()
    rec = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    out, _ = prune_clashing_leaves(mol, rec, ["C"] * len(rec))
    assert len(Chem.GetMolFrags(out)) == 1
    Chem.SanitizeMol(out)  # raises if the pruning left a bad valence


def test_it_never_prunes_below_the_floor() -> None:
    """Burying the whole molecule still leaves a molecule."""
    mol = _mol("CCCCCCCCO")
    conf = mol.GetConformer()
    rec = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    out, _ = prune_clashing_leaves(mol, rec, ["C"] * len(rec))
    assert out.GetNumAtoms() >= 6


def test_the_conformer_follows_the_deletion() -> None:
    """Coordinates must be re-read after each removal, not cached."""
    mol = _mol()
    conf = mol.GetConformer()
    before = [np.array(list(conf.GetAtomPosition(i))) for i in range(mol.GetNumAtoms())]
    leaf = next(a.GetIdx() for a in mol.GetAtoms() if a.GetDegree() == 1)
    rec, els = _receptor_on(np.array(list(conf.GetAtomPosition(leaf))))
    out, n = prune_clashing_leaves(mol, rec, els)
    if n:
        oc = out.GetConformer()
        assert oc.GetNumAtoms() == out.GetNumAtoms()
        pts = [list(oc.GetAtomPosition(i)) for i in range(out.GetNumAtoms())]
        assert np.isfinite(pts).all()
    assert mol.GetNumAtoms() == len(before)  # the caller's molecule is not mutated
    conf2 = mol.GetConformer()
    still = [
        np.array(list(conf2.GetAtomPosition(i))) for i in range(mol.GetNumAtoms())
    ]
    assert all(np.allclose(a, b) for a, b in zip(before, still, strict=True))


@pytest.mark.parametrize("element", ["C", "N", "O"])
def test_the_clash_threshold_is_the_deployed_one(element: str) -> None:
    """An atom just outside CLASH_FRACTION of the summed radii is not a clash."""
    mol = _mol()
    conf = mol.GetConformer()
    leaf = next(a.GetIdx() for a in mol.GetAtoms() if a.GetDegree() == 1)
    pos = np.array(list(conf.GetAtomPosition(leaf)))
    summed = float(
        vdw_radii([mol.GetAtomWithIdx(leaf).GetSymbol()])[0] + vdw_radii([element])[0]
    )
    just_outside = pos + np.array([CLASH_FRACTION * summed * 1.01, 0.0, 0.0])
    _, n = prune_clashing_leaves(mol, np.array([just_outside]), [element])
    assert n == 0


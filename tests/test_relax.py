"""The relaxation must repair geometry without touching the molecule.

Two properties matter and they pull against each other: the conformer has to
get better, and the pose has to stay put. A relaxation that quietly moved the
ligand out of its pocket would improve every geometry metric and invalidate
every docking score, and nothing downstream would notice.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from prolit.chem.relax import relax_local_geometry

TOLERANCE = 0.102  # the tokenizer's measured coordinate MAE


def _embedded(smiles: str, seed: int = 0xC0FFEE) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    return Chem.RemoveHs(mol)


def _heavy_coords(mol: Chem.Mol) -> np.ndarray:
    return mol.GetConformer().GetPositions()


def _jitter(mol: Chem.Mol, sigma: float, seed: int = 7) -> Chem.Mol:
    """Displace every atom by Gaussian noise, as quantisation error would."""
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=(out.GetNumAtoms(), 3))
    for i, (x, y, z) in enumerate(_heavy_coords(out) + noise):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    return out


def test_the_molecule_is_unchanged() -> None:
    """Relaxation moves atoms; it must never re-bond or re-element them."""
    mol = _embedded("c1ccccc1C(=O)Nc1ccncc1")
    relaxed = relax_local_geometry(_jitter(mol, 0.15), TOLERANCE)
    assert relaxed is not None
    assert Chem.MolToSmiles(relaxed) == Chem.MolToSmiles(mol)


def test_the_pose_stays_within_the_tolerance() -> None:
    """The flat bottom is the whole safety argument: check it holds.

    Not an exact bound -- the restraint is flat-bottomed rather than hard, so an
    atom can be pushed past it by a stiff MMFF term -- but it must stay the same
    order as the tolerance rather than drifting free.
    """
    mol = _embedded("CCc1ccc(cc1)S(=O)(=O)NC1CC1")
    start = _jitter(mol, 0.15)
    relaxed = relax_local_geometry(start, TOLERANCE)
    assert relaxed is not None
    shift = np.linalg.norm(_heavy_coords(relaxed) - _heavy_coords(start), axis=1)
    assert shift.mean() < 2 * TOLERANCE
    assert shift.max() < 10 * TOLERANCE


def test_a_distorted_conformer_gets_better() -> None:
    """The point of the exercise: bond lengths move back toward the clean ones."""
    mol = _embedded("c1ccccc1C(=O)Nc1ccncc1")
    noisy = _jitter(mol, 0.2)

    def bond_error(candidate: Chem.Mol) -> float:
        clean, other = mol.GetConformer(), candidate.GetConformer()
        errors = [
            abs(
                np.linalg.norm(
                    np.array(other.GetAtomPosition(b.GetBeginAtomIdx()))
                    - np.array(other.GetAtomPosition(b.GetEndAtomIdx()))
                )
                - np.linalg.norm(
                    np.array(clean.GetAtomPosition(b.GetBeginAtomIdx()))
                    - np.array(clean.GetAtomPosition(b.GetEndAtomIdx()))
                )
            )
            for b in mol.GetBonds()
        ]
        return float(np.mean(errors))

    relaxed = relax_local_geometry(noisy, TOLERANCE)
    assert relaxed is not None
    assert bond_error(relaxed) < bond_error(noisy)


def test_a_wider_tolerance_moves_the_pose_further() -> None:
    """The parameter has to actually be the thing that bounds the motion."""
    start = _jitter(_embedded("c1ccccc1C(=O)Nc1ccncc1"), 0.2)
    shifts = []
    for tolerance in (0.05, 0.5):
        relaxed = relax_local_geometry(start, tolerance)
        assert relaxed is not None
        shifts.append(
            float(
                np.linalg.norm(
                    _heavy_coords(relaxed) - _heavy_coords(start), axis=1
                ).mean()
            )
        )
    assert shifts[0] < shifts[1]


@pytest.mark.parametrize("smiles", ["[Se]", "[U]"])
def test_untypeable_molecules_return_none(smiles: str) -> None:
    """MMFF cannot type everything. The caller keeps the original in that case,
    so this must fail by returning None rather than by raising."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        pytest.skip(f"RDKit will not parse {smiles}")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (0.0, 0.0, float(i)))
    mol.AddConformer(conf)
    assert relax_local_geometry(mol, TOLERANCE) is None

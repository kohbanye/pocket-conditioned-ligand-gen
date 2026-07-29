"""Tests for GEOM ligand-only pretraining pieces.

Covers ``random_rotation_matrix`` (a proper rotation: orthogonal, det +1), the
molecule-level split helper, and the RDKit-mol extraction.

The other half of the rotation-augmentation contract -- that
``rotate_atom_descriptor`` is *equivalent* to recomputing the descriptor in the
rotated frame, which is what lets the tokenizer run RDKit once per conformer and
synthesise K orientations -- lives in ``test_atom_descriptor.py``.
"""

from __future__ import annotations

import numpy as np

from src.data.geom import _rd_mol_to_atoms_bonds, assign_split
from src.tokenizers.geometry import (
    random_rotation_matrix,
)


def _make_molecule() -> tuple[list, list]:
    """A small, non-degenerate heavy-atom skeleton with bonds."""
    atoms = [
        ("C", 1.2, 0.3, -0.5),
        ("C", 2.1, -0.7, 0.4),
        ("O", 0.1, 1.4, 0.9),
        ("N", -1.3, 0.2, -0.7),
        ("C", -2.0, -1.1, 0.6),
        ("C", 3.4, 0.9, 1.1),
    ]
    bonds = [(0, 1, 1), (0, 2, 1), (0, 3, 1), (3, 4, 1), (1, 5, 1)]
    return atoms, bonds


def _heavy_centroid(atoms: list) -> np.ndarray:
    coords = np.array([(a[1], a[2], a[3]) for a in atoms], dtype=np.float64)
    return coords.mean(axis=0)


class TestRandomRotation:
    def test_is_proper_rotation(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            r = random_rotation_matrix(rng)
            assert r.shape == (3, 3)
            # Orthogonal: R R^T = I
            assert np.allclose(r @ r.T, np.eye(3), atol=1e-10)
            # Proper (no reflection): det = +1
            assert np.isclose(np.linalg.det(r), 1.0, atol=1e-10)

    def test_distinct_samples(self) -> None:
        rng = np.random.default_rng(1)
        a = random_rotation_matrix(rng)
        b = random_rotation_matrix(rng)
        assert not np.allclose(a, b)


class TestAssignSplit:
    def test_deterministic(self) -> None:
        assert assign_split("CCO", 0.1, 0.1) == assign_split("CCO", 0.1, 0.1)

    def test_distribution(self) -> None:
        smiles = [f"C{'C' * (i % 7)}O{i}" for i in range(4000)]
        splits = [assign_split(s, 0.1, 0.1) for s in smiles]
        frac_test = splits.count("test") / len(splits)
        frac_val = splits.count("val") / len(splits)
        assert 0.06 < frac_test < 0.14
        assert 0.06 < frac_val < 0.14


class TestRdMolExtraction:
    def test_extracts_atoms_and_bonds(self) -> None:
        from rdkit import Chem  # noqa: PLC0415
        from rdkit.Chem import AllChem  # noqa: PLC0415

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        assert AllChem.EmbedMolecule(mol, randomSeed=1) == 0
        parsed = _rd_mol_to_atoms_bonds(mol)
        assert parsed is not None
        elements = {a[0] for a in parsed["atoms"]}
        assert {"C", "O"} <= elements
        assert len(parsed["bonds"]) > 0
        # bond codes are SDF integers in {1,2,3,4}
        assert all(1 <= b[2] <= 4 for b in parsed["bonds"])

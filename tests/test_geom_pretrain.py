"""Tests for GEOM ligand-only pretraining pieces.

Covers the two correctness-critical pieces of the rotation-augmentation path:

1. ``random_rotation_matrix`` is a proper rotation (orthogonal, det +1).
2. ``rotate_ligand_descriptor`` (the cheap augmentation primitive) is
   *equivalent* to recomputing the descriptor in the rotated frame -- this is
   what lets the tokenizer run RDKit once per conformer and synthesise K
   orientations.

Plus the molecule-level split helper and the RDKit-mol extraction.
"""

from __future__ import annotations

import numpy as np

from src.data.geom import _rd_mol_to_atoms_bonds, assign_split
from src.tokenizers.descriptor_schema import (
    K_NEIGHBORS,
    LIGAND_LAYOUT,
    fields_by_name,
)
from src.tokenizers.geometry import (
    random_rotation_matrix,
    spherical_to_cartesian_np,
)
from src.tokenizers.ligand import LigandDescriptor, rotate_ligand_descriptor


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


class TestRotateDescriptorEquivalence:
    """rotate_ligand_descriptor == recompute in the rotated frame."""

    def test_equivalence(self) -> None:
        atoms, bonds = _make_molecule()
        centroid = _heavy_centroid(atoms)
        rot = random_rotation_matrix(np.random.default_rng(7))
        desc = LigandDescriptor()

        base, _, _ = desc.compute(atoms, bonds, pocket_frame=(centroid, np.eye(3)))
        direct, _, _ = desc.compute(atoms, bonds, pocket_frame=(centroid, rot))
        helper = rotate_ligand_descriptor(base, rot)

        f = fields_by_name(LIGAND_LAYOUT)

        # Categorical / element slots are rotation-invariant -> identical.
        for name in (
            "element",
            "charge",
            "hybrid",
            "aromatic",
            "ring",
            "numH",
            "knn_elements",
        ):
            sl = slice(f[name].start, f[name].end)
            assert np.array_equal(helper[:, sl], direct[:, sl]), name

        # Compare orientation-dependent slots in Cartesian space (avoids angle
        # wrap-around ambiguity at the poles).
        def _coord_cart(arr: np.ndarray, start: int) -> np.ndarray:
            return spherical_to_cartesian_np(arr[:, start : start + 4])

        np.testing.assert_allclose(
            _coord_cart(helper, f["coord"].start),
            _coord_cart(direct, f["coord"].start),
            atol=1e-4,
        )
        for k in range(K_NEIGHBORS):
            s = f["knn_offsets"].start + 4 * k
            np.testing.assert_allclose(
                _coord_cart(helper, s), _coord_cart(direct, s), atol=1e-4
            )

    def test_radius_is_rotation_invariant(self) -> None:
        atoms, bonds = _make_molecule()
        centroid = _heavy_centroid(atoms)
        rot = random_rotation_matrix(np.random.default_rng(3))
        desc = LigandDescriptor()
        base, _, _ = desc.compute(atoms, bonds, pocket_frame=(centroid, np.eye(3)))
        direct, _, _ = desc.compute(atoms, bonds, pocket_frame=(centroid, rot))
        f = fields_by_name(LIGAND_LAYOUT)
        # coord r (column 0 of the coord slot) is invariant under rotation.
        np.testing.assert_allclose(
            base[:, f["coord"].start], direct[:, f["coord"].start], atol=1e-5
        )


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

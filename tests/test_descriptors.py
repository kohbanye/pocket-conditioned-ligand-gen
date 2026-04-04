"""Tests for SE(3) invariance of protein and ligand descriptors."""

import numpy as np
from scipy.spatial.transform import Rotation

from src.tokenizers.ligand import SE3InvariantDescriptor
from src.tokenizers.protein import ProteinBackboneDescriptor


def _random_rotation() -> np.ndarray:
    """Generate a random 3x3 rotation matrix."""
    return Rotation.random().as_matrix()


def _apply_rigid_transform(
    coords: np.ndarray, rotation: np.ndarray, translation: np.ndarray,
) -> np.ndarray:
    """Apply rotation + translation to coordinates."""
    return (rotation @ coords.T).T + translation


# --- Protein backbone descriptor tests ---


def _make_helix_backbone(num_residues: int = 20) -> np.ndarray:
    """Generate synthetic alpha-helix-like backbone coordinates.

    Returns shape (num_residues, 3, 3) for (N, CA, C).
    """
    coords = np.zeros((num_residues, 3, 3), dtype=np.float64)
    for i in range(num_residues):
        t = i * 1.5  # 1.5 Å rise per residue (roughly)
        angle = i * np.radians(100)  # ~100° rotation per residue
        r = 2.3  # helix radius

        # CA position along helix
        ca = np.array([r * np.cos(angle), r * np.sin(angle), t])

        # N offset from CA
        n_offset = np.array([0.47, -0.26, -1.0])
        n = ca + n_offset

        # C offset from CA
        c_offset = np.array([-0.47, 0.26, 0.5])
        c = ca + c_offset

        coords[i] = [n, ca, c]

    return coords


class TestProteinBackboneDescriptor:
    def test_se3_invariance_rotation(self) -> None:
        """Descriptors should be identical under rotation."""
        backbone = _make_helix_backbone(20)
        desc_computer = ProteinBackboneDescriptor(num_neighbors=8)

        original = desc_computer.compute(backbone)

        rot = _random_rotation()
        rotated = np.zeros_like(backbone)
        for i in range(len(backbone)):
            for j in range(3):
                rotated[i, j] = rot @ backbone[i, j]

        rotated_desc = desc_computer.compute(rotated)

        np.testing.assert_allclose(original, rotated_desc, atol=1e-5)

    def test_se3_invariance_translation(self) -> None:
        """Descriptors should be identical under translation."""
        backbone = _make_helix_backbone(20)
        desc_computer = ProteinBackboneDescriptor(num_neighbors=8)

        original = desc_computer.compute(backbone)

        translation = np.array([10.0, -5.0, 3.0])
        translated = backbone + translation

        translated_desc = desc_computer.compute(translated)

        np.testing.assert_allclose(original, translated_desc, atol=1e-5)

    def test_se3_invariance_full(self) -> None:
        """Descriptors should be identical under rotation + translation."""
        backbone = _make_helix_backbone(30)
        desc_computer = ProteinBackboneDescriptor(num_neighbors=16)

        original = desc_computer.compute(backbone)

        rot = _random_rotation()
        trans = np.array([100.0, -50.0, 25.0])
        transformed = np.zeros_like(backbone)
        for i in range(len(backbone)):
            for j in range(3):
                transformed[i, j] = rot @ backbone[i, j] + trans

        transformed_desc = desc_computer.compute(transformed)

        np.testing.assert_allclose(original, transformed_desc, atol=1e-5)

    def test_output_shape(self) -> None:
        """Check descriptor output dimensions."""
        backbone = _make_helix_backbone(10)
        k = 8
        desc_computer = ProteinBackboneDescriptor(num_neighbors=k)
        result = desc_computer.compute(backbone)

        assert result.shape == (10, k + 4)

    def test_small_protein(self) -> None:
        """Handle protein with fewer residues than k."""
        backbone = _make_helix_backbone(3)
        desc_computer = ProteinBackboneDescriptor(num_neighbors=16)
        result = desc_computer.compute(backbone)

        assert result.shape == (3, 20)


# --- Ligand descriptor tests ---


def _make_ethanol() -> tuple[
    list[tuple[str, float, float, float]], list[tuple[int, int, int]],
]:
    """Create ethanol molecule (C2H5OH) with approximate 3D coords."""
    atoms: list[tuple[str, float, float, float]] = [
        ("C", 0.0, 0.0, 0.0),
        ("C", 1.54, 0.0, 0.0),
        ("O", 2.31, 1.26, 0.0),
        ("H", -0.36, 1.03, 0.0),
        ("H", -0.36, -0.51, 0.89),
        ("H", -0.36, -0.51, -0.89),
        ("H", 1.90, -0.51, 0.89),
        ("H", 1.90, -0.51, -0.89),
        ("H", 3.27, 1.05, 0.0),
    ]
    bonds: list[tuple[int, int, int]] = [
        (0, 1, 1),
        (1, 2, 1),
        (0, 3, 1),
        (0, 4, 1),
        (0, 5, 1),
        (1, 6, 1),
        (1, 7, 1),
        (2, 8, 1),
    ]
    return atoms, bonds


class TestLigandSE3Descriptor:
    def test_se3_invariance_rotation(self) -> None:
        """Descriptors should be identical under rotation."""
        atoms, bonds = _make_ethanol()
        desc_computer = SE3InvariantDescriptor(max_neighbors=4)

        original, _elements = desc_computer.compute(atoms, bonds)

        rot = _random_rotation()
        coords = np.array([(a[1], a[2], a[3]) for a in atoms])
        rotated_coords = _apply_rigid_transform(coords, rot, np.zeros(3))
        rotated_atoms = [
            (a[0], *rotated_coords[i].tolist()) for i, a in enumerate(atoms)
        ]

        rotated_desc, _ = desc_computer.compute(rotated_atoms, bonds)

        np.testing.assert_allclose(original, rotated_desc, atol=1e-5)

    def test_se3_invariance_translation(self) -> None:
        """Descriptors should be identical under translation."""
        atoms, bonds = _make_ethanol()
        desc_computer = SE3InvariantDescriptor(max_neighbors=4)

        original, _ = desc_computer.compute(atoms, bonds)

        trans = np.array([10.0, -5.0, 3.0])
        translated_atoms: list[tuple[str, float, float, float]] = [
            (a[0], a[1] + trans[0], a[2] + trans[1], a[3] + trans[2]) for a in atoms
        ]

        translated_desc, _ = desc_computer.compute(translated_atoms, bonds)

        np.testing.assert_allclose(original, translated_desc, atol=1e-5)

    def test_se3_invariance_full(self) -> None:
        """Descriptors should be identical under rotation + translation."""
        atoms, bonds = _make_ethanol()
        desc_computer = SE3InvariantDescriptor(max_neighbors=4)

        original, _ = desc_computer.compute(atoms, bonds)

        rot = _random_rotation()
        trans = np.array([100.0, -50.0, 25.0])
        coords = np.array([(a[1], a[2], a[3]) for a in atoms])
        transformed_coords = _apply_rigid_transform(coords, rot, trans)
        transformed_atoms = [
            (a[0], *transformed_coords[i].tolist()) for i, a in enumerate(atoms)
        ]

        transformed_desc, _ = desc_computer.compute(transformed_atoms, bonds)

        np.testing.assert_allclose(original, transformed_desc, atol=1e-5)

    def test_output_shape(self) -> None:
        """Check descriptor output dimensions."""
        atoms, bonds = _make_ethanol()
        desc_computer = SE3InvariantDescriptor(max_neighbors=4)
        result, elements = desc_computer.compute(atoms, bonds)

        assert result.shape == (9, 14)
        assert len(elements) == 9
        assert elements[0] == "C"
        assert elements[2] == "O"

    def test_single_atom(self) -> None:
        """Handle single-atom molecule."""
        atoms: list[tuple[str, float, float, float]] = [("Fe", 0.0, 0.0, 0.0)]
        bonds: list[tuple[int, int, int]] = []
        desc_computer = SE3InvariantDescriptor(max_neighbors=4)
        result, elements = desc_computer.compute(atoms, bonds)

        assert result.shape == (1, 14)
        assert elements == ["Fe"]

    def test_empty_molecule(self) -> None:
        """Handle empty molecule."""
        desc_computer = SE3InvariantDescriptor(max_neighbors=4)
        result, elements = desc_computer.compute([], [])

        assert result.shape == (0, 14)
        assert elements == []

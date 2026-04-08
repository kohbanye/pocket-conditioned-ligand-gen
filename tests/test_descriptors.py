"""Tests for invertible ligand and protein descriptors.

Validates:
- Round-trip reconstruction (descriptor → coords → descriptor match)
- SE(3) invariance of descriptors
- Edge cases (single atom, small molecules, small pockets)
"""

import numpy as np
from scipy.spatial.transform import Rotation

from src.tokenizers.ligand import LigandDescriptor
from src.tokenizers.protein import (
    PocketDescriptor,
    _compute_canonical_frame,
)


def _random_rotation() -> np.ndarray:
    """Generate a random 3x3 rotation matrix."""
    return Rotation.random().as_matrix()


def _apply_rigid_transform(
    coords: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Apply rotation + translation to coordinates."""
    return (rotation @ coords.T).T + translation


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Compute RMSD between two point sets after optimal superposition."""
    # Center both
    a_centered = a - a.mean(axis=0)
    b_centered = b - b.mean(axis=0)

    # Kabsch alignment
    h = a_centered.T @ b_centered
    u, _s, vt = np.linalg.svd(h)
    d = np.linalg.det(vt.T @ u.T)
    sign_matrix = np.diag([1, 1, d])
    rotation = vt.T @ sign_matrix @ u.T
    a_aligned = a_centered @ rotation.T

    return float(np.sqrt(np.mean((a_aligned - b_centered) ** 2)))


# ---------------------------------------------------------------------------
# Protein backbone descriptor tests
# ---------------------------------------------------------------------------


def _make_asymmetric_backbone(num_residues: int = 20) -> np.ndarray:
    """Generate backbone with well-separated PCA eigenvalues.

    Uses different scales along each axis to avoid PCA degeneracy.
    Returns shape (num_residues, 3, 3) for (N, CA, C).
    """
    coords = np.zeros((num_residues, 3, 3), dtype=np.float64)
    for i in range(num_residues):
        # Large variance in x, medium in y, small in z
        ca = np.array(
            [
                i * 3.0,
                np.sin(i * 0.5) * 2.0,
                np.cos(i * 0.7) * 0.5,
            ]
        )
        n = ca + np.array([0.47, -0.26, -1.0])
        c = ca + np.array([-0.47, 0.26, 0.5])

        coords[i] = [n, ca, c]

    return coords


class TestPocketDescriptor:
    def test_round_trip_exact(self) -> None:
        """descriptor → backbone coords should be exact with metadata."""
        backbone = _make_asymmetric_backbone(20)
        desc = PocketDescriptor()

        descriptors, metadata = desc.compute(backbone)
        reconstructed = PocketDescriptor.descriptor_to_backbone_coords(
            descriptors,
            metadata,
        )

        np.testing.assert_allclose(reconstructed, backbone, atol=1e-4)

    def test_se3_invariance_rotation(self) -> None:
        """Descriptors should be identical under rotation."""
        backbone = _make_asymmetric_backbone(20)
        desc = PocketDescriptor()

        original, _ = desc.compute(backbone)

        rot = _random_rotation()
        rotated = np.zeros_like(backbone)
        for i in range(len(backbone)):
            for j in range(3):
                rotated[i, j] = rot @ backbone[i, j]

        rotated_desc, _ = desc.compute(rotated)

        np.testing.assert_allclose(original, rotated_desc, atol=1e-4)

    def test_se3_invariance_translation(self) -> None:
        """Descriptors should be identical under translation."""
        backbone = _make_asymmetric_backbone(20)
        desc = PocketDescriptor()

        original, _ = desc.compute(backbone)

        translation = np.array([10.0, -5.0, 3.0])
        translated = backbone + translation

        translated_desc, _ = desc.compute(translated)

        np.testing.assert_allclose(original, translated_desc, atol=1e-4)

    def test_se3_invariance_full(self) -> None:
        """Descriptors should be identical under rotation + translation."""
        backbone = _make_asymmetric_backbone(30)
        desc = PocketDescriptor()

        original, _ = desc.compute(backbone)

        rot = _random_rotation()
        trans = np.array([100.0, -50.0, 25.0])
        transformed = np.zeros_like(backbone)
        for i in range(len(backbone)):
            for j in range(3):
                transformed[i, j] = rot @ backbone[i, j] + trans

        transformed_desc, _ = desc.compute(transformed)

        np.testing.assert_allclose(original, transformed_desc, atol=1e-4)

    def test_output_shape(self) -> None:
        """Check descriptor output dimensions."""
        backbone = _make_asymmetric_backbone(10)
        desc = PocketDescriptor()
        result, _ = desc.compute(backbone)

        assert result.shape == (10, 9)

    def test_small_pocket(self) -> None:
        """Handle pocket with very few residues."""
        backbone = _make_asymmetric_backbone(2)
        desc = PocketDescriptor()
        result, metadata = desc.compute(backbone)

        assert result.shape == (2, 9)

        # Round-trip should still work
        reconstructed = PocketDescriptor.descriptor_to_backbone_coords(
            result,
            metadata,
        )
        np.testing.assert_allclose(reconstructed, backbone, atol=1e-4)

    def test_single_residue(self) -> None:
        """Handle single-residue pocket."""
        backbone = _make_asymmetric_backbone(1)
        desc = PocketDescriptor()
        result, metadata = desc.compute(backbone)

        assert result.shape == (1, 9)

        reconstructed = PocketDescriptor.descriptor_to_backbone_coords(
            result,
            metadata,
        )
        np.testing.assert_allclose(reconstructed, backbone, atol=1e-4)


# ---------------------------------------------------------------------------
# Ligand descriptor tests
# ---------------------------------------------------------------------------


def _make_ethanol() -> tuple[
    list[tuple[str, float, float, float]],
    list[tuple[int, int, int]],
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


def _make_cyclopropane() -> tuple[
    list[tuple[str, float, float, float]],
    list[tuple[int, int, int]],
]:
    """Create cyclopropane (C3H6) with approximate 3D coords — has a ring."""
    r = 0.87  # C-C bond ~ 1.51 Å, equilateral triangle radius
    atoms: list[tuple[str, float, float, float]] = [
        ("C", r, 0.0, 0.0),
        ("C", r * np.cos(2 * np.pi / 3), r * np.sin(2 * np.pi / 3), 0.0),
        ("C", r * np.cos(4 * np.pi / 3), r * np.sin(4 * np.pi / 3), 0.0),
        ("H", r + 0.5, 0.5, 0.5),
        ("H", r + 0.5, -0.5, -0.5),
        ("H", -1.0, 1.2, 0.5),
        ("H", -1.0, 1.2, -0.5),
        ("H", -1.0, -1.2, 0.5),
        ("H", -1.0, -1.2, -0.5),
    ]
    bonds: list[tuple[int, int, int]] = [
        (0, 1, 1),
        (1, 2, 1),
        (0, 2, 1),  # ring closure
        (0, 3, 1),
        (0, 4, 1),
        (1, 5, 1),
        (1, 6, 1),
        (2, 7, 1),
        (2, 8, 1),
    ]
    return atoms, bonds


class TestLigandDescriptor:
    def test_round_trip_ethanol(self) -> None:
        """descriptor → coords should match original (up to SE(3))."""
        atoms, bonds = _make_ethanol()
        desc = LigandDescriptor()

        descriptors, _elements, metadata = desc.compute(atoms, bonds)
        reconstructed = LigandDescriptor.descriptor_to_coords(
            descriptors,
            metadata,
        )

        original_coords = np.array([(a[1], a[2], a[3]) for a in atoms])
        rmsd = _kabsch_rmsd(original_coords, reconstructed)
        assert rmsd < 1e-6, f"Round-trip RMSD = {rmsd}"

    def test_round_trip_cyclopropane(self) -> None:
        """Round-trip with a ring-containing molecule (spanning tree only)."""
        atoms, bonds = _make_cyclopropane()
        desc = LigandDescriptor()

        descriptors, _elements, metadata = desc.compute(atoms, bonds)
        reconstructed = LigandDescriptor.descriptor_to_coords(
            descriptors,
            metadata,
        )

        original_coords = np.array([(a[1], a[2], a[3]) for a in atoms])
        rmsd = _kabsch_rmsd(original_coords, reconstructed)
        assert rmsd < 1e-6, f"Round-trip RMSD = {rmsd}"

    def test_ring_closures_detected(self) -> None:
        """Ring closure bonds should appear in metadata."""
        atoms, bonds = _make_cyclopropane()
        desc = LigandDescriptor()

        _descriptors, _elements, metadata = desc.compute(atoms, bonds)
        assert len(metadata["ring_closures"]) >= 1

    def test_se3_invariance_rotation(self) -> None:
        """Descriptors should be identical under rotation."""
        atoms, bonds = _make_ethanol()
        desc = LigandDescriptor()

        original, _, _ = desc.compute(atoms, bonds)

        rot = _random_rotation()
        coords = np.array([(a[1], a[2], a[3]) for a in atoms])
        rotated_coords = _apply_rigid_transform(coords, rot, np.zeros(3))
        rotated_atoms = [
            (a[0], *rotated_coords[i].tolist()) for i, a in enumerate(atoms)
        ]

        rotated_desc, _, _ = desc.compute(rotated_atoms, bonds)

        np.testing.assert_allclose(original, rotated_desc, atol=1e-5)

    def test_se3_invariance_translation(self) -> None:
        """Descriptors should be identical under translation."""
        atoms, bonds = _make_ethanol()
        desc = LigandDescriptor()

        original, _, _ = desc.compute(atoms, bonds)

        trans = np.array([10.0, -5.0, 3.0])
        translated_atoms: list[tuple[str, float, float, float]] = [
            (a[0], a[1] + trans[0], a[2] + trans[1], a[3] + trans[2]) for a in atoms
        ]

        translated_desc, _, _ = desc.compute(translated_atoms, bonds)

        np.testing.assert_allclose(original, translated_desc, atol=1e-5)

    def test_se3_invariance_full(self) -> None:
        """Descriptors should be identical under rotation + translation."""
        atoms, bonds = _make_ethanol()
        desc = LigandDescriptor()

        original, _, _ = desc.compute(atoms, bonds)

        rot = _random_rotation()
        trans = np.array([100.0, -50.0, 25.0])
        coords = np.array([(a[1], a[2], a[3]) for a in atoms])
        transformed_coords = _apply_rigid_transform(coords, rot, trans)
        transformed_atoms = [
            (a[0], *transformed_coords[i].tolist()) for i, a in enumerate(atoms)
        ]

        transformed_desc, _, _ = desc.compute(transformed_atoms, bonds)

        np.testing.assert_allclose(original, transformed_desc, atol=1e-5)

    def test_output_shape(self) -> None:
        """Check descriptor output dimensions."""
        atoms, bonds = _make_ethanol()
        desc = LigandDescriptor()
        result, elements, metadata = desc.compute(atoms, bonds)

        assert result.shape == (9, 4)
        assert len(elements) == 9
        assert "order" in metadata
        assert "refs" in metadata
        assert "ring_closures" in metadata

    def test_single_atom(self) -> None:
        """Handle single-atom molecule."""
        atoms: list[tuple[str, float, float, float]] = [("Fe", 0.0, 0.0, 0.0)]
        bonds: list[tuple[int, int, int]] = []
        desc = LigandDescriptor()
        result, elements, _ = desc.compute(atoms, bonds)

        assert result.shape == (1, 4)
        assert elements == ["Fe"]

    def test_two_atoms(self) -> None:
        """Handle two-atom molecule (e.g. HCl)."""
        atoms: list[tuple[str, float, float, float]] = [
            ("H", 0.0, 0.0, 0.0),
            ("Cl", 1.27, 0.0, 0.0),
        ]
        bonds: list[tuple[int, int, int]] = [(0, 1, 1)]
        desc = LigandDescriptor()
        result, elements, metadata = desc.compute(atoms, bonds)

        assert result.shape == (2, 4)
        assert len(elements) == 2

        # Round-trip
        reconstructed = LigandDescriptor.descriptor_to_coords(
            result,
            metadata,
        )
        original_coords = np.array([(a[1], a[2], a[3]) for a in atoms])
        # For 2 atoms, Kabsch alignment is degenerate — check distance instead
        orig_dist = np.linalg.norm(original_coords[0] - original_coords[1])
        recon_dist = np.linalg.norm(reconstructed[0] - reconstructed[1])
        assert abs(orig_dist - recon_dist) < 1e-6

    def test_empty_molecule(self) -> None:
        """Handle empty molecule."""
        desc = LigandDescriptor()
        result, elements, _ = desc.compute([], [])

        assert result.shape == (0, 4)
        assert elements == []


# ---------------------------------------------------------------------------
# Pocket-anchored ligand descriptor tests
# ---------------------------------------------------------------------------


def _make_pocket_frame() -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic pocket canonical frame for testing."""
    backbone = _make_asymmetric_backbone(10)
    ca_coords = backbone[:, 1]
    return _compute_canonical_frame(ca_coords)


class TestAnchoredLigandDescriptor:
    def test_round_trip_ethanol_anchored(self) -> None:
        """Anchored descriptor → coords (global) should match original."""
        atoms, bonds = _make_ethanol()
        pocket_frame = _make_pocket_frame()
        desc = LigandDescriptor()

        descriptors, _elems, metadata = desc.compute(
            atoms,
            bonds,
            pocket_frame=pocket_frame,
        )
        assert metadata["anchored"] is True

        reconstructed = LigandDescriptor.descriptor_to_coords(
            descriptors,
            metadata,
            pocket_frame=pocket_frame,
        )

        original_coords = np.array([(a[1], a[2], a[3]) for a in atoms])
        np.testing.assert_allclose(reconstructed, original_coords, atol=1e-5)

    def test_round_trip_cyclopropane_anchored(self) -> None:
        """Anchored round-trip with a ring molecule."""
        atoms, bonds = _make_cyclopropane()
        pocket_frame = _make_pocket_frame()
        desc = LigandDescriptor()

        descriptors, _elems, metadata = desc.compute(
            atoms,
            bonds,
            pocket_frame=pocket_frame,
        )
        reconstructed = LigandDescriptor.descriptor_to_coords(
            descriptors,
            metadata,
            pocket_frame=pocket_frame,
        )

        original_coords = np.array([(a[1], a[2], a[3]) for a in atoms])
        np.testing.assert_allclose(reconstructed, original_coords, atol=1e-5)

    def test_se3_invariance_anchored(self) -> None:
        """Rotating the whole complex should give the same descriptors."""
        atoms, bonds = _make_ethanol()
        backbone = _make_asymmetric_backbone(10)
        prot_desc = PocketDescriptor()
        desc = LigandDescriptor()

        # Original
        _, prot_meta = prot_desc.compute(backbone)
        frame_orig = (prot_meta["centroid"], prot_meta["rotation"])
        desc_orig, _, _ = desc.compute(atoms, bonds, pocket_frame=frame_orig)

        # Rotate everything together
        rot = _random_rotation()
        trans = np.array([50.0, -30.0, 10.0])

        rotated_backbone = np.zeros_like(backbone)
        for i in range(len(backbone)):
            for j in range(3):
                rotated_backbone[i, j] = rot @ backbone[i, j] + trans

        coords = np.array([(a[1], a[2], a[3]) for a in atoms])
        rotated_coords = _apply_rigid_transform(coords, rot, trans)
        rotated_atoms = [
            (a[0], *rotated_coords[i].tolist()) for i, a in enumerate(atoms)
        ]

        _, prot_meta_rot = prot_desc.compute(rotated_backbone)
        frame_rot = (prot_meta_rot["centroid"], prot_meta_rot["rotation"])
        desc_rot, _, _ = desc.compute(rotated_atoms, bonds, pocket_frame=frame_rot)

        np.testing.assert_allclose(desc_orig, desc_rot, atol=1e-4)

    def test_anchored_root_encodes_position(self) -> None:
        """Root atom descriptor should differ from non-anchored padding."""
        atoms, bonds = _make_ethanol()
        pocket_frame = _make_pocket_frame()
        desc = LigandDescriptor()

        anchored, _, _ = desc.compute(atoms, bonds, pocket_frame=pocket_frame)
        standalone, _, _ = desc.compute(atoms, bonds)

        # Root (pos 0): anchored should encode spherical position, not padding
        assert not np.allclose(anchored[0], standalone[0])

    def test_fallback_without_pocket_frame(self) -> None:
        """Without pocket_frame, behavior matches original standalone mode."""
        atoms, bonds = _make_ethanol()
        desc = LigandDescriptor()

        result, _, metadata = desc.compute(atoms, bonds)
        assert metadata["anchored"] is False
        # Root should be padding
        np.testing.assert_allclose(result[0], [0, 0, 0, 1], atol=1e-7)

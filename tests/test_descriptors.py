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
    BackboneZMatrixDescriptor,
    PocketDescriptor,
    _compute_canonical_frame,
    _detect_segments,
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


# ---------------------------------------------------------------------------
# Segment detection tests
# ---------------------------------------------------------------------------


class TestDetectSegments:
    def test_single_segment(self) -> None:
        """All contiguous residues should form one segment."""
        ids = [("A", 10), ("A", 11), ("A", 12), ("A", 13)]
        assert _detect_segments(ids) == [(0, 4)]

    def test_multiple_segments(self) -> None:
        """Gaps in residue indices should split into segments."""
        ids = [("A", 10), ("A", 11), ("A", 20), ("A", 21), ("A", 22)]
        assert _detect_segments(ids) == [(0, 2), (2, 5)]

    def test_different_chains(self) -> None:
        """Different chains should be separate segments."""
        ids = [("A", 10), ("A", 11), ("B", 10), ("B", 11)]
        assert _detect_segments(ids) == [(0, 2), (2, 4)]

    def test_single_residue_segments(self) -> None:
        """Isolated residues should each be their own segment."""
        ids = [("A", 10), ("A", 20), ("A", 30)]
        assert _detect_segments(ids) == [(0, 1), (1, 2), (2, 3)]

    def test_empty(self) -> None:
        assert _detect_segments([]) == []


# ---------------------------------------------------------------------------
# Backbone Z-matrix descriptor tests
# ---------------------------------------------------------------------------


def _make_realistic_backbone(num_residues: int = 10) -> np.ndarray:
    """Generate a realistic protein backbone using ideal geometry.

    Uses standard backbone bond lengths and angles with varying
    phi/psi torsion angles to create a plausible chain.
    """
    # Ideal backbone geometry
    d_n_ca = 1.458  # N-CA bond
    d_ca_c = 1.525  # CA-C bond
    d_c_n = 1.329   # C-N peptide bond
    angle_ca_c_n = np.radians(116.2)
    angle_c_n_ca = np.radians(121.7)
    angle_n_ca_c = np.radians(111.2)

    coords = np.zeros((num_residues, 3, 3), dtype=np.float64)

    # Place first residue at origin
    coords[0, 0] = [0.0, 0.0, 0.0]  # N
    coords[0, 1] = [d_n_ca, 0.0, 0.0]  # CA
    # C: placed at bond angle from N-CA
    coords[0, 2] = coords[0, 1] + d_ca_c * np.array([
        -np.cos(np.pi - angle_n_ca_c),
        np.sin(np.pi - angle_n_ca_c),
        0.0,
    ])

    rng = np.random.default_rng(42)
    for i in range(1, num_residues):
        prev_n = coords[i - 1, 0]
        prev_ca = coords[i - 1, 1]
        prev_c = coords[i - 1, 2]

        # Phi, psi, omega torsion angles (random but realistic)
        psi_prev = rng.uniform(-np.pi, np.pi)
        omega = np.pi + rng.normal(0, 0.05)  # ~180 deg (trans)
        phi = rng.uniform(-np.pi, np.pi)

        # Place N(i) using NeRF: dihedral N(i-1)-CA(i-1)-C(i-1)-N(i)
        from src.tokenizers.geometry import place_atom  # noqa: PLC0415
        n_pos = place_atom(
            prev_n, prev_ca, prev_c, d_c_n, angle_ca_c_n, psi_prev,
        )
        coords[i, 0] = n_pos

        # Place CA(i): dihedral CA(i-1)-C(i-1)-N(i)-CA(i)
        ca_pos = place_atom(
            prev_ca, prev_c, n_pos, d_n_ca, angle_c_n_ca, omega,
        )
        coords[i, 1] = ca_pos

        # Place C(i): dihedral C(i-1)-N(i)-CA(i)-C(i)
        c_pos = place_atom(
            prev_c, n_pos, ca_pos, d_ca_c, angle_n_ca_c, phi,
        )
        coords[i, 2] = c_pos

    return coords


class TestBackboneZMatrixDescriptor:
    def test_round_trip_single_segment(self) -> None:
        """descriptor → backbone coords should round-trip for contiguous residues."""
        backbone = _make_realistic_backbone(10)
        residue_ids = [("A", i) for i in range(10)]
        desc = BackboneZMatrixDescriptor()

        descriptors, metadata = desc.compute(backbone, residue_ids)
        reconstructed = BackboneZMatrixDescriptor.descriptor_to_backbone_coords(
            descriptors, metadata,
        )

        np.testing.assert_allclose(reconstructed, backbone, atol=1e-4)

    def test_round_trip_multiple_segments(self) -> None:
        """Round-trip with non-contiguous residues (multiple segments)."""
        backbone = _make_realistic_backbone(10)
        # Simulate non-contiguous: residues 0-3 and 7-9 (gap at 4-6)
        residue_ids = [
            ("A", 10), ("A", 11), ("A", 12), ("A", 13),
            ("A", 20), ("A", 21), ("A", 22),
            ("B", 5), ("B", 6), ("B", 7),
        ]
        desc = BackboneZMatrixDescriptor()

        descriptors, metadata = desc.compute(backbone, residue_ids)
        reconstructed = BackboneZMatrixDescriptor.descriptor_to_backbone_coords(
            descriptors, metadata,
        )

        np.testing.assert_allclose(reconstructed, backbone, atol=1e-4)

    def test_se3_invariance_rotation(self) -> None:
        """Descriptors should be identical under rotation."""
        backbone = _make_realistic_backbone(8)
        residue_ids = [("A", i) for i in range(8)]
        desc = BackboneZMatrixDescriptor()

        original, _ = desc.compute(backbone, residue_ids)

        rot = _random_rotation()
        rotated = np.zeros_like(backbone)
        for i in range(len(backbone)):
            for j in range(3):
                rotated[i, j] = rot @ backbone[i, j]

        rotated_desc, _ = desc.compute(rotated, residue_ids)

        np.testing.assert_allclose(original, rotated_desc, atol=1e-4)

    def test_se3_invariance_translation(self) -> None:
        """Descriptors should be identical under translation."""
        backbone = _make_realistic_backbone(8)
        residue_ids = [("A", i) for i in range(8)]
        desc = BackboneZMatrixDescriptor()

        original, _ = desc.compute(backbone, residue_ids)

        translation = np.array([10.0, -5.0, 3.0])
        translated = backbone + translation

        translated_desc, _ = desc.compute(translated, residue_ids)

        np.testing.assert_allclose(original, translated_desc, atol=1e-4)

    def test_se3_invariance_full(self) -> None:
        """Descriptors should be identical under rotation + translation."""
        backbone = _make_realistic_backbone(8)
        residue_ids = [("A", i) for i in range(8)]
        desc = BackboneZMatrixDescriptor()

        original, _ = desc.compute(backbone, residue_ids)

        rot = _random_rotation()
        trans = np.array([100.0, -50.0, 25.0])
        transformed = np.zeros_like(backbone)
        for i in range(len(backbone)):
            for j in range(3):
                transformed[i, j] = rot @ backbone[i, j] + trans

        transformed_desc, _ = desc.compute(transformed, residue_ids)

        np.testing.assert_allclose(original, transformed_desc, atol=1e-4)

    def test_output_shape(self) -> None:
        """Check descriptor output dimensions."""
        backbone = _make_realistic_backbone(5)
        residue_ids = [("A", i) for i in range(5)]
        desc = BackboneZMatrixDescriptor()
        result, metadata = desc.compute(backbone, residue_ids)

        assert result.shape == (5, 12)
        assert "centroid" in metadata
        assert "rotation" in metadata
        assert "segments" in metadata

    def test_single_residue(self) -> None:
        """Handle single-residue pocket."""
        backbone = _make_realistic_backbone(1)
        residue_ids = [("A", 1)]
        desc = BackboneZMatrixDescriptor()
        result, metadata = desc.compute(backbone, residue_ids)

        assert result.shape == (1, 12)

        reconstructed = BackboneZMatrixDescriptor.descriptor_to_backbone_coords(
            result, metadata,
        )
        np.testing.assert_allclose(reconstructed, backbone, atol=1e-4)

    def test_two_residues(self) -> None:
        """Handle two-residue pocket."""
        backbone = _make_realistic_backbone(2)
        residue_ids = [("A", 1), ("A", 2)]
        desc = BackboneZMatrixDescriptor()
        result, metadata = desc.compute(backbone, residue_ids)

        assert result.shape == (2, 12)

        reconstructed = BackboneZMatrixDescriptor.descriptor_to_backbone_coords(
            result, metadata,
        )
        np.testing.assert_allclose(reconstructed, backbone, atol=1e-4)

    def test_with_explicit_pocket_frame(self) -> None:
        """compute() should accept an explicit pocket_frame."""
        backbone = _make_realistic_backbone(5)
        residue_ids = [("A", i) for i in range(5)]
        ca = backbone[:, 1]
        centroid, rotation = _compute_canonical_frame(ca)
        pocket_frame = (centroid, rotation)

        desc = BackboneZMatrixDescriptor()
        result, metadata = desc.compute(
            backbone, residue_ids, pocket_frame=pocket_frame,
        )

        reconstructed = BackboneZMatrixDescriptor.descriptor_to_backbone_coords(
            result, metadata,
        )
        np.testing.assert_allclose(reconstructed, backbone, atol=1e-4)

    def test_descriptor_values_bounded(self) -> None:
        """Bond lengths, angles, sin/cos should be in expected ranges."""
        backbone = _make_realistic_backbone(10)
        residue_ids = [("A", i) for i in range(10)]
        desc = BackboneZMatrixDescriptor()
        result, metadata = desc.compute(backbone, residue_ids)

        # For continuation residues (skip first segment start)
        segments = metadata["segments"]
        for seg_start, seg_end in segments:
            for idx in range(seg_start + 1, seg_end):
                for atom_offset in range(3):
                    base = atom_offset * 4
                    bond_len = result[idx, base]
                    bond_angle = result[idx, base + 1]
                    sin_tor = result[idx, base + 2]
                    cos_tor = result[idx, base + 3]

                    # Bond lengths should be ~1.3-1.6 Angstrom
                    assert 1.0 < bond_len < 2.0, (
                        f"Bond length {bond_len} out of range"
                    )
                    # Bond angles should be reasonable (~1.5-2.5 rad)
                    assert 1.0 < bond_angle < 3.0, (
                        f"Bond angle {bond_angle} out of range"
                    )
                    # sin/cos should be in [-1, 1]
                    assert -1.01 < sin_tor < 1.01
                    assert -1.01 < cos_tor < 1.01

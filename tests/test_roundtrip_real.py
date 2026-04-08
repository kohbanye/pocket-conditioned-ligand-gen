"""Round-trip tests with realistic 3D structures.

Uses RDKit to generate drug-like molecules with proper 3D conformations
and realistic protein backbone geometries to verify that the full
descriptor pipeline preserves atomic coordinates.
"""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation

from src.tokenizers.ligand import LigandDescriptor
from src.tokenizers.protein import PocketDescriptor

# ---------------------------------------------------------------------------
# Helpers: molecule generation from SMILES
# ---------------------------------------------------------------------------

_DRUG_MOLECULES: dict[str, str] = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "benzene": "c1ccccc1",
    "cyclohexane": "C1CCCCC1",
    "naphthalene": "c1ccc2ccccc2c1",
    "alanine_dipeptide": "CC(NC(C)=O)C(=O)NC",
    "penicillin_g": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
}


def _mol_from_smiles(
    smiles: str,
    *,
    add_hs: bool = True,
) -> tuple[list[tuple[str, float, float, float]], list[tuple[int, int, int]]]:
    """Generate 3D coordinates for a SMILES string via RDKit.

    Returns (atoms, bonds) in the format expected by
    :class:`LigandDescriptor`.
    """
    mol = Chem.MolFromSmiles(smiles)
    if add_hs:
        mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, randomSeed=42)
    if result != 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    atoms: list[tuple[str, float, float, float]] = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        atoms.append((sym, pos.x, pos.y, pos.z))

    bonds: list[tuple[int, int, int]] = []
    for bond in mol.GetBonds():
        bt = bond.GetBondTypeAsDouble()
        bt_int = round(bt) if bt <= 3 else 1
        bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bt_int))

    return atoms, bonds


# ---------------------------------------------------------------------------
# Helpers: protein backbone generation
# ---------------------------------------------------------------------------


def _make_realistic_backbone(num_residues: int = 30) -> np.ndarray:
    """Generate backbone coordinates with ideal bond geometry.

    Uses standard backbone bond lengths and angles to build a
    polypeptide chain with varied phi/psi angles (mix of helix and
    sheet regions).

    Returns shape ``(num_residues, 3, 3)`` for (N, CA, C).
    """
    # Ideal bond lengths (Angstroms)
    d_n_ca = 1.458
    d_ca_c = 1.523
    d_c_n = 1.329  # peptide bond

    # Ideal bond angles (radians)
    ang_ca_c_n = np.radians(116.2)
    ang_c_n_ca = np.radians(121.7)
    ang_n_ca_c = np.radians(111.2)

    # Varied phi/psi to create mixed secondary structure
    rng = np.random.default_rng(42)
    phi_psi = []
    for i in range(num_residues):
        if i % 3 == 0:
            # Alpha helix region
            phi_psi.append((np.radians(-57), np.radians(-47)))
        elif i % 3 == 1:
            # Beta sheet region
            phi_psi.append((np.radians(-120), np.radians(130)))
        else:
            # Random coil
            phi_psi.append((rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)))

    # Build backbone iteratively using NeRF-like placement
    all_positions: list[np.ndarray] = []

    # First residue: place at origin with standard geometry
    n0 = np.array([0.0, 0.0, 0.0])
    ca0 = np.array([d_n_ca, 0.0, 0.0])
    c0 = ca0 + d_ca_c * np.array(
        [np.cos(np.pi - ang_n_ca_c), np.sin(np.pi - ang_n_ca_c), 0.0],
    )
    all_positions.extend([n0, ca0, c0])

    for i in range(1, num_residues):
        phi, psi = phi_psi[i]

        # Previous atoms
        prev_n = all_positions[-3]
        prev_ca = all_positions[-2]
        prev_c = all_positions[-1]

        # Place N(i) from C(i-1)-CA(i-1)-N(i-1) + peptide bond
        ref_for_n = (
            all_positions[-4]
            if len(all_positions) >= 4
            else prev_n - np.array([1, 0, 0])
        )
        n_i = _nerf_place(
            ref_for_n,
            prev_ca,
            prev_c,
            d_c_n,
            ang_ca_c_n,
            psi if i > 1 else 0.0,
        )

        # Place CA(i) from N(i)
        ca_i = _nerf_place(prev_ca, prev_c, n_i, d_n_ca, ang_c_n_ca, phi)

        # Place C(i) from CA(i)
        c_i = _nerf_place(prev_c, n_i, ca_i, d_ca_c, ang_n_ca_c, psi)

        all_positions.extend([n_i, ca_i, c_i])

    return np.array(all_positions, dtype=np.float64).reshape(-1, 3, 3)


def _nerf_place(  # noqa: PLR0913
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bond_length: float,
    bond_angle: float,
    torsion: float,
) -> np.ndarray:
    """Place atom D using Natural Extension Reference Frame."""
    bc = c - b
    bc_norm = np.linalg.norm(bc)
    if bc_norm < 1e-8:
        return c + np.array([bond_length, 0.0, 0.0])
    bc_hat = bc / bc_norm

    ab = b - a
    n = np.cross(bc_hat, ab)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-8:
        perp = (
            np.array([0.0, 1.0, 0.0])
            if abs(bc_hat[1]) < 0.9
            else np.array([1.0, 0.0, 0.0])
        )
        n = np.cross(bc_hat, perp)
        n = n / np.linalg.norm(n)
    else:
        n = n / n_norm

    m = np.cross(n, bc_hat)

    return c + bond_length * (
        np.cos(bond_angle) * bc_hat
        + np.sin(bond_angle) * np.cos(torsion) * m
        + np.sin(bond_angle) * np.sin(torsion) * n
    )


def _extract_pocket_subset(
    backbone: np.ndarray,
    indices: list[int],
) -> np.ndarray:
    """Extract non-contiguous pocket residues (simulates real pocket)."""
    return backbone[indices]


# ---------------------------------------------------------------------------
# Pairwise distance check (stronger than RMSD for verifying geometry)
# ---------------------------------------------------------------------------


def _max_pairwise_distance_error(a: np.ndarray, b: np.ndarray) -> float:
    """Compute max absolute difference in pairwise distances."""
    da = pdist(a)
    db = pdist(b)
    return float(np.max(np.abs(da - db)))


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD after optimal superposition."""
    ac = a - a.mean(axis=0)
    bc = b - b.mean(axis=0)
    h = ac.T @ bc
    u, _s, vt = np.linalg.svd(h)
    d = np.linalg.det(vt.T @ u.T)
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return float(np.sqrt(np.mean((ac @ r.T - bc) ** 2)))


# ===========================================================================
# Ligand round-trip tests with real molecules
# ===========================================================================


class TestLigandRoundTripReal:
    """Round-trip reconstruction of RDKit-generated drug molecules."""

    @pytest.mark.parametrize(("name", "smiles"), list(_DRUG_MOLECULES.items()))
    def test_standalone_round_trip(self, name: str, smiles: str) -> None:
        """Z-matrix round-trip preserves all pairwise distances."""
        atoms, bonds = _mol_from_smiles(smiles)
        desc = LigandDescriptor()

        descriptors, _elements, metadata = desc.compute(atoms, bonds)
        reconstructed = desc.descriptor_to_coords(descriptors, metadata)

        original = np.array([(a[1], a[2], a[3]) for a in atoms])
        max_err = _max_pairwise_distance_error(original, reconstructed)
        assert max_err < 1e-5, f"{name}: max pairwise distance error = {max_err}"

    @pytest.mark.parametrize(("name", "smiles"), list(_DRUG_MOLECULES.items()))
    def test_anchored_round_trip(self, name: str, smiles: str) -> None:
        """Anchored round-trip recovers exact global coordinates."""
        atoms, bonds = _mol_from_smiles(smiles)
        backbone = _make_realistic_backbone(15)
        pocket_backbone = _extract_pocket_subset(
            backbone,
            [0, 2, 4, 7, 9, 11, 13],
        )

        prot_desc = PocketDescriptor()
        _, prot_meta = prot_desc.compute(pocket_backbone)
        pocket_frame = (prot_meta["centroid"], prot_meta["rotation"])

        lig_desc = LigandDescriptor()
        descriptors, _elems, metadata = lig_desc.compute(
            atoms,
            bonds,
            pocket_frame=pocket_frame,
        )
        reconstructed = lig_desc.descriptor_to_coords(
            descriptors,
            metadata,
            pocket_frame=pocket_frame,
        )

        original = np.array([(a[1], a[2], a[3]) for a in atoms])
        np.testing.assert_allclose(
            reconstructed,
            original,
            atol=1e-5,
            err_msg=f"{name}: anchored round-trip failed",
        )

    @pytest.mark.parametrize(("name", "smiles"), list(_DRUG_MOLECULES.items()))
    def test_descriptor_shape(self, name: str, smiles: str) -> None:
        """All molecules produce (N, 4) descriptors."""
        atoms, bonds = _mol_from_smiles(smiles)
        desc = LigandDescriptor()
        result, elements, _ = desc.compute(atoms, bonds)

        assert result.shape == (len(atoms), 4), f"{name}: wrong shape"
        assert len(elements) == len(atoms), f"{name}: element count mismatch"

    def test_heavy_atoms_only(self) -> None:
        """Round-trip works without hydrogens (heavy atoms only)."""
        atoms, bonds = _mol_from_smiles(
            _DRUG_MOLECULES["ibuprofen"],
            add_hs=False,
        )
        desc = LigandDescriptor()
        descriptors, _elems, metadata = desc.compute(atoms, bonds)
        reconstructed = desc.descriptor_to_coords(descriptors, metadata)

        original = np.array([(a[1], a[2], a[3]) for a in atoms])
        rmsd = _kabsch_rmsd(original, reconstructed)
        assert rmsd < 1e-5, f"Heavy-atom-only RMSD = {rmsd}"


# ===========================================================================
# Protein backbone round-trip tests with realistic geometry
# ===========================================================================


class TestProteinRoundTripReal:
    """Round-trip reconstruction of realistic protein backbone."""

    def test_full_chain_round_trip(self) -> None:
        """30-residue chain: exact round-trip."""
        backbone = _make_realistic_backbone(30)
        desc = PocketDescriptor()

        descriptors, metadata = desc.compute(backbone)
        reconstructed = desc.descriptor_to_backbone_coords(descriptors, metadata)

        np.testing.assert_allclose(reconstructed, backbone, atol=1e-4)

    def test_noncontiguous_pocket_round_trip(self) -> None:
        """Non-contiguous pocket subset: exact round-trip."""
        backbone = _make_realistic_backbone(30)
        pocket_indices = [1, 3, 7, 8, 12, 15, 20, 25, 28]
        pocket = _extract_pocket_subset(backbone, pocket_indices)

        desc = PocketDescriptor()
        descriptors, metadata = desc.compute(pocket)
        reconstructed = desc.descriptor_to_backbone_coords(descriptors, metadata)

        np.testing.assert_allclose(reconstructed, pocket, atol=1e-4)

    def test_backbone_bond_lengths_preserved(self) -> None:
        """N-CA and CA-C bond lengths should be preserved after round-trip."""
        backbone = _make_realistic_backbone(20)
        desc = PocketDescriptor()

        descriptors, metadata = desc.compute(backbone)
        recon = desc.descriptor_to_backbone_coords(descriptors, metadata)

        for i in range(len(backbone)):
            # N-CA bond
            orig_n_ca = np.linalg.norm(backbone[i, 1] - backbone[i, 0])
            recon_n_ca = np.linalg.norm(recon[i, 1] - recon[i, 0])
            assert abs(orig_n_ca - recon_n_ca) < 1e-4, f"Residue {i}: N-CA mismatch"

            # CA-C bond
            orig_ca_c = np.linalg.norm(backbone[i, 2] - backbone[i, 1])
            recon_ca_c = np.linalg.norm(recon[i, 2] - recon[i, 1])
            assert abs(orig_ca_c - recon_ca_c) < 1e-4, f"Residue {i}: CA-C mismatch"

    def test_ca_pairwise_distances_preserved(self) -> None:
        """All CA-CA pairwise distances must be preserved."""
        backbone = _make_realistic_backbone(20)
        desc = PocketDescriptor()

        descriptors, metadata = desc.compute(backbone)
        recon = desc.descriptor_to_backbone_coords(descriptors, metadata)

        orig_ca = backbone[:, 1]
        recon_ca = recon[:, 1]
        max_err = _max_pairwise_distance_error(orig_ca, recon_ca)
        assert max_err < 1e-4, f"CA pairwise distance error = {max_err}"


# ===========================================================================
# Full complex (pocket + ligand) integration test
# ===========================================================================


class TestComplexIntegration:
    """End-to-end test: pocket + ligand in shared coordinate frame."""

    def test_complex_round_trip(self) -> None:
        """Full pipeline: pocket frame → anchored ligand → reconstruct both."""
        # Build a realistic pocket
        backbone = _make_realistic_backbone(25)
        pocket_indices = [0, 2, 5, 8, 10, 13, 16, 19, 22]
        pocket = _extract_pocket_subset(backbone, pocket_indices)

        # Protein descriptor round-trip
        prot_desc_computer = PocketDescriptor()
        prot_descriptors, prot_meta = prot_desc_computer.compute(pocket)
        pocket_frame = (prot_meta["centroid"], prot_meta["rotation"])

        prot_recon = prot_desc_computer.descriptor_to_backbone_coords(
            prot_descriptors,
            prot_meta,
        )
        np.testing.assert_allclose(prot_recon, pocket, atol=1e-4)

        # Generate a ligand near the pocket
        atoms, bonds = _mol_from_smiles(_DRUG_MOLECULES["aspirin"])

        # Ligand descriptor round-trip (anchored)
        lig_desc_computer = LigandDescriptor()
        lig_descriptors, _elems, lig_meta = lig_desc_computer.compute(
            atoms,
            bonds,
            pocket_frame=pocket_frame,
        )
        lig_recon = lig_desc_computer.descriptor_to_coords(
            lig_descriptors,
            lig_meta,
            pocket_frame=pocket_frame,
        )

        original_lig = np.array([(a[1], a[2], a[3]) for a in atoms])
        np.testing.assert_allclose(lig_recon, original_lig, atol=1e-5)

        # Verify that reconstructed ligand and pocket are in the same
        # global coordinate frame by checking that their centroids are
        # at plausible relative positions (not at the origin)
        lig_centroid = lig_recon.mean(axis=0)
        pocket_centroid = prot_recon[:, 1].mean(axis=0)  # CA centroid
        separation = np.linalg.norm(lig_centroid - pocket_centroid)
        assert separation > 0.1, "Ligand and pocket should not both be at origin"

    def test_se3_invariance_complex(self) -> None:
        """Rotating the entire complex produces identical descriptors."""
        backbone = _make_realistic_backbone(15)
        pocket = _extract_pocket_subset(backbone, [0, 3, 6, 9, 12])
        atoms, bonds = _mol_from_smiles(_DRUG_MOLECULES["caffeine"])

        prot_computer = PocketDescriptor()
        lig_computer = LigandDescriptor()

        # Original
        _, meta_orig = prot_computer.compute(pocket)
        frame_orig = (meta_orig["centroid"], meta_orig["rotation"])
        lig_desc_orig, _, _ = lig_computer.compute(
            atoms,
            bonds,
            pocket_frame=frame_orig,
        )

        # Apply random rigid transform to everything
        rot = Rotation.random(random_state=123).as_matrix()
        trans = np.array([50.0, -25.0, 10.0])

        rot_pocket = np.zeros_like(pocket)
        for i in range(len(pocket)):
            for j in range(3):
                rot_pocket[i, j] = rot @ pocket[i, j] + trans

        coords = np.array([(a[1], a[2], a[3]) for a in atoms])
        rot_coords = (rot @ coords.T).T + trans
        rot_atoms = [(a[0], *rot_coords[i].tolist()) for i, a in enumerate(atoms)]

        _, meta_rot = prot_computer.compute(rot_pocket)
        frame_rot = (meta_rot["centroid"], meta_rot["rotation"])
        lig_desc_rot, _, _ = lig_computer.compute(
            rot_atoms,
            bonds,
            pocket_frame=frame_rot,
        )

        np.testing.assert_allclose(lig_desc_orig, lig_desc_rot, atol=1e-4)

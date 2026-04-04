"""Protein pocket structure and sequence tokenization.

Provides:
- Pocket extraction from PDB files
- SE(3)-invariant backbone descriptor (k-NN based)
- VQ-VAE for structure tokenization
- Simple amino acid sequence tokenizer
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from torch import Tensor, nn

from src.tokenizers.codebook import EMACodebook

if TYPE_CHECKING:
    from pathlib import Path

    from src.config import PocketExtractionConfig, ProteinVQVAEConfig

_EPS = 1e-8

# Standard amino acid 3-letter to 1-letter mapping
AA_3TO1: dict[str, str] = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

BACKBONE_ATOMS = ("N", "CA", "C")


def extract_pocket(
    pdb_path: str | Path,
    ligand_coords: np.ndarray,
    config: PocketExtractionConfig,
) -> tuple[np.ndarray, str] | None:
    """Extract pocket residues near the ligand from a PDB file.

    Args:
        pdb_path: Path to the receptor PDB file.
        ligand_coords: Ligand heavy-atom coordinates, shape ``(N_lig, 3)``.
        config: Pocket extraction parameters.

    Returns:
        Tuple of (backbone_coords, pocket_sequence) where backbone_coords has
        shape ``(L, 3, 3)`` for (N, CA, C) and pocket_sequence is a string of
        1-letter amino acid codes.  Returns ``None`` if extraction fails.
    """
    from Bio.PDB import PDBParser  # noqa: PLC0415

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", str(pdb_path))
    model = structure[0]

    residues_with_dist: list[tuple[float, object]] = []
    for chain in model:
        for residue in chain:
            resname = residue.get_resname()
            if resname not in AA_3TO1:
                continue
            # Check that backbone atoms exist
            if not all(atom in residue for atom in BACKBONE_ATOMS):
                continue
            ca_coord = residue["CA"].get_vector().get_array()
            min_dist = np.min(np.linalg.norm(ligand_coords - ca_coord, axis=1))
            if min_dist <= config.distance_cutoff:
                residues_with_dist.append((min_dist, residue))

    if not residues_with_dist:
        return None

    # Sort by distance and truncate
    residues_with_dist.sort(key=lambda x: x[0])
    selected = [r for _, r in residues_with_dist[: config.max_residues]]

    # Sort back by residue index for consistent ordering
    selected.sort(key=lambda r: (r.get_parent().id, r.get_id()[1]))

    backbone_coords = []
    pocket_seq = []
    for residue in selected:
        coords = [residue[atom].get_vector().get_array() for atom in BACKBONE_ATOMS]
        backbone_coords.append(coords)
        pocket_seq.append(AA_3TO1[residue.get_resname()])

    return np.array(backbone_coords, dtype=np.float32), "".join(pocket_seq)


def extract_full_sequence(pdb_path: str | Path) -> str:
    """Extract full amino acid sequence from a PDB file.

    Args:
        pdb_path: Path to the receptor PDB file.

    Returns:
        1-letter amino acid sequence string.
    """
    from Bio.PDB import PDBParser  # noqa: PLC0415

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", str(pdb_path))
    model = structure[0]

    residues = []
    for chain in model:
        for residue in chain:
            resname = residue.get_resname()
            if resname in AA_3TO1:
                residues.append((chain.id, residue.get_id()[1], resname))

    residues.sort(key=lambda x: (x[0], x[1]))
    return "".join(AA_3TO1[r[2]] for r in residues)


class ProteinBackboneDescriptor:
    """Compute SE(3)-invariant per-residue descriptor from backbone atoms.

    Uses sorted CA-CA distances to k nearest neighbors plus backbone
    dihedral angles.  All features are SE(3)-invariant by construction
    (distances are norms; dihedrals use relative vectors).

    Descriptor dimensions: k + 4.
    """

    def __init__(self, num_neighbors: int = 16) -> None:
        self.num_neighbors = num_neighbors

    def compute(self, backbone_coords: np.ndarray) -> np.ndarray:
        """Compute descriptors for all residues.

        Args:
            backbone_coords: Shape ``(L, 3, 3)`` — (N, CA, C) per residue.

        Returns:
            Descriptors of shape ``(L, k+4)``.
        """
        n_coords = backbone_coords[:, 0]  # (L, 3)
        ca_coords = backbone_coords[:, 1]  # (L, 3)
        c_coords = backbone_coords[:, 2]  # (L, 3)

        # Compute sorted k-NN distances (SE(3)-invariant)
        knn_dists = self._compute_knn_distances(ca_coords)

        # Compute backbone dihedrals
        dihedrals = self._compute_dihedrals(n_coords, ca_coords, c_coords)

        return np.concatenate([knn_dists, dihedrals], axis=1).astype(np.float32)

    def _compute_knn_distances(self, ca_coords: np.ndarray) -> np.ndarray:
        """Compute sorted distances to k nearest CA neighbors.

        Returns:
            Sorted distances of shape ``(L, k)``.
        """
        num_residues = len(ca_coords)
        k = self.num_neighbors

        # Pairwise CA-CA distances: (L, L)
        diff = ca_coords[:, None, :] - ca_coords[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)

        features = np.zeros((num_residues, k), dtype=np.float32)
        for i in range(num_residues):
            actual_k = min(k, num_residues - 1)
            # Get k smallest distances, sorted
            kth_dists = np.partition(dists[i], actual_k)[:actual_k]
            kth_dists.sort()
            features[i, :actual_k] = kth_dists

        return features

    def _compute_dihedrals(
        self,
        n_coords: np.ndarray,
        ca_coords: np.ndarray,
        c_coords: np.ndarray,
    ) -> np.ndarray:
        """Compute backbone phi/psi dihedrals as sin/cos.

        Returns:
            Shape ``(L, 4)`` — [sin(phi), cos(phi), sin(psi), cos(psi)].
        """
        num_residues = len(n_coords)
        dihedrals = np.zeros((num_residues, 4), dtype=np.float32)

        for i in range(num_residues):
            if i > 0:  # phi: C(i-1)-N(i)-CA(i)-C(i)
                phi = self._dihedral_angle(
                    c_coords[i - 1], n_coords[i], ca_coords[i], c_coords[i],
                )
                dihedrals[i, 0] = np.sin(phi)
                dihedrals[i, 1] = np.cos(phi)

            if i < num_residues - 1:  # psi: N(i)-CA(i)-C(i)-N(i+1)
                psi = self._dihedral_angle(
                    n_coords[i], ca_coords[i], c_coords[i], n_coords[i + 1],
                )
                dihedrals[i, 2] = np.sin(psi)
                dihedrals[i, 3] = np.cos(psi)

        return dihedrals

    @staticmethod
    def _dihedral_angle(
        p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
    ) -> float:
        """Compute dihedral angle defined by four points."""
        b1 = p1 - p0
        b2 = p2 - p1
        b3 = p3 - p2

        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        if n1_norm < _EPS or n2_norm < _EPS:
            return 0.0

        n1 = n1 / n1_norm
        n2 = n2 / n2_norm

        b2_unit = b2 / (np.linalg.norm(b2) + _EPS)
        m = np.cross(n1, b2_unit)

        x = np.dot(n1, n2)
        y = np.dot(m, n2)

        return float(np.arctan2(y, x))


class ProteinStructureVQVAE(nn.Module):
    """VQ-VAE for protein backbone structure tokenization.

    Encodes per-residue SE(3)-invariant descriptors into discrete codes.
    """

    def __init__(self, config: ProteinVQVAEConfig) -> None:
        super().__init__()
        self.config = config
        descriptor_dim = config.num_neighbors + 4

        self.encoder = nn.Sequential(
            nn.Linear(descriptor_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, descriptor_dim),
        )

        self.codebook = EMACodebook(
            num_codes=config.codebook_size,
            code_dim=config.latent_dim,
            ema_decay=config.ema_decay,
            commitment_cost=config.commitment_cost,
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass: encode, quantize, decode.

        Args:
            x: Descriptors of shape ``(B, descriptor_dim)``.

        Returns:
            Dict with keys: reconstructed, indices, commitment_loss,
            reconstruction_loss.
        """
        z = self.encoder(x)
        quantized, indices, commitment_loss = self.codebook(z)
        reconstructed = self.decoder(quantized)
        reconstruction_loss = (x - reconstructed).pow(2).mean()

        return {
            "reconstructed": reconstructed,
            "indices": indices,
            "commitment_loss": commitment_loss,
            "reconstruction_loss": reconstruction_loss,
        }

    def encode(self, x: Tensor) -> Tensor:
        """Encode descriptors to codebook indices."""
        z = self.encoder(x)
        _, indices, _ = self.codebook(z)
        return indices

    def decode(self, indices: Tensor) -> Tensor:
        """Decode codebook indices back to descriptors."""
        quantized = self.codebook.lookup(indices)
        return self.decoder(quantized)


class ProteinSequenceTokenizer:
    """Simple amino acid sequence tokenizer (no learning required)."""

    VOCAB: ClassVar[list[str]] = [*"ACDEFGHIKLMNPQRSTVWY", "X"]

    def __init__(self) -> None:
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.VOCAB)}

    def encode(self, sequence: str) -> list[str]:
        """Convert sequence string to list of single-character tokens."""
        return [c if c in self.aa_to_idx else "X" for c in sequence]

    @property
    def vocab_size(self) -> int:
        return len(self.VOCAB)

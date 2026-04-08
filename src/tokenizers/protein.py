"""Protein pocket structure and sequence tokenization.

Provides:
- Pocket extraction from PDB files
- PCA canonical-frame per-residue descriptor (9D, invertible)
- VQ-VAE for structure tokenization
- Simple amino acid sequence tokenizer
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

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


# ---------------------------------------------------------------------------
# PCA canonical frame helpers
# ---------------------------------------------------------------------------


def _compute_canonical_frame(
    ca_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a deterministic canonical frame from CA coordinates via PCA.

    Returns:
        centroid: ``(3,)`` mean of CA positions.
        rotation: ``(3, 3)`` orthogonal matrix (rows = principal-component
            directions, sign-disambiguated).
    """
    centroid = ca_coords.mean(axis=0).astype(np.float64)
    centered = (ca_coords - centroid).astype(np.float64)

    if len(ca_coords) < 2:  # noqa: PLR2004
        return centroid, np.eye(3, dtype=np.float64)

    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    # When fewer than 3 residues, SVD returns fewer rows in Vt;
    # pad to a full 3x3 orthogonal matrix.
    if vt.shape[0] < 3:  # noqa: PLR2004
        vt_full = np.eye(3, dtype=np.float64)
        vt_full[: vt.shape[0]] = vt
        # Make the missing rows orthogonal to the existing ones
        if vt.shape[0] == 1:
            # Pick two perpendicular vectors
            v0 = vt_full[0]
            perp = (
                np.array([0.0, 0.0, 1.0])
                if abs(v0[2]) < 0.9  # noqa: PLR2004
                else np.array([1.0, 0.0, 0.0])
            )
            vt_full[1] = np.cross(v0, perp)
            vt_full[1] /= np.linalg.norm(vt_full[1])
            vt_full[2] = np.cross(v0, vt_full[1])
        elif vt.shape[0] == 2:  # noqa: PLR2004
            vt_full[2] = np.cross(vt_full[0], vt_full[1])
        vt = vt_full

    # Sign disambiguation: flip each axis so the atom with the largest
    # absolute projection determines the sign (more robust than sum).
    for i in range(3):
        proj = centered @ vt[i]
        max_idx = int(np.argmax(np.abs(proj)))
        if proj[max_idx] < 0:
            vt[i] *= -1

    # Ensure right-handed frame
    if np.linalg.det(vt) < 0:
        vt[2] *= -1

    return centroid, vt.astype(np.float64)


# ---------------------------------------------------------------------------
# Descriptor class
# ---------------------------------------------------------------------------


class PocketDescriptor:
    """PCA canonical-frame per-residue descriptor with exact inverse.

    Each residue receives a **9-D** descriptor:

    *   CA position in canonical frame (3D)
    *   N-CA offset in canonical frame (3D)
    *   C-CA offset in canonical frame (3D)

    The canonical frame is derived from PCA of the pocket CA coordinates with
    sign disambiguation, ensuring SE(3)-invariant descriptors.
    """

    DESCRIPTOR_DIM = 9

    def compute(
        self,
        backbone_coords: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Compute descriptors for all residues.

        Args:
            backbone_coords: Shape ``(L, 3, 3)`` — (N, CA, C) per residue.

        Returns:
            descriptors: ``(L, 9)`` float32 array.
            metadata: ``{'centroid', 'rotation'}`` for inverse transform.
        """
        n_coords = backbone_coords[:, 0].astype(np.float64)  # (L, 3)
        ca_coords = backbone_coords[:, 1].astype(np.float64)  # (L, 3)
        c_coords = backbone_coords[:, 2].astype(np.float64)  # (L, 3)

        centroid, rotation = _compute_canonical_frame(ca_coords)

        # Transform into canonical frame:  x_can = (x - centroid) @ V
        #   where V = rotation.T  (rotation rows are PC directions)
        v = rotation.T
        ca_can = (ca_coords - centroid) @ v
        n_can = (n_coords - centroid) @ v
        c_can = (c_coords - centroid) @ v

        n_offset = n_can - ca_can
        c_offset = c_can - ca_can

        descriptors = np.concatenate([ca_can, n_offset, c_offset], axis=1)

        metadata: dict[str, Any] = {
            "centroid": centroid,
            "rotation": rotation,
        }
        return descriptors.astype(np.float32), metadata

    # ---- inverse transform ------------------------------------------------

    @staticmethod
    def descriptor_to_backbone_coords(
        descriptors: np.ndarray,
        metadata: dict[str, Any],
    ) -> np.ndarray:
        """Reconstruct backbone ``(N, CA, C)`` from descriptors + metadata.

        The round-trip is exact (up to floating-point precision).

        Returns:
            backbone_coords: ``(L, 3, 3)`` float32 array.
        """
        centroid: np.ndarray = metadata["centroid"]
        rotation: np.ndarray = metadata["rotation"]  # (3, 3), rows = PCs

        ca_can = descriptors[:, :3].astype(np.float64)
        n_offset = descriptors[:, 3:6].astype(np.float64)
        c_offset = descriptors[:, 6:9].astype(np.float64)

        n_can = ca_can + n_offset
        c_can = ca_can + c_offset

        # Inverse:  x_global = x_can @ V^T + centroid = x_can @ rotation + centroid
        #   because V = rotation.T  →  V^T = rotation
        ca_global = ca_can @ rotation + centroid
        n_global = n_can @ rotation + centroid
        c_global = c_can @ rotation + centroid

        num_residues = len(descriptors)
        backbone = np.zeros((num_residues, 3, 3), dtype=np.float64)
        backbone[:, 0] = n_global
        backbone[:, 1] = ca_global
        backbone[:, 2] = c_global

        return backbone.astype(np.float32)


# ---------------------------------------------------------------------------
# VQ-VAE
# ---------------------------------------------------------------------------


class ProteinStructureVQVAE(nn.Module):
    """VQ-VAE for protein backbone structure tokenization.

    Encodes per-residue descriptors into discrete codes.
    """

    def __init__(self, config: ProteinVQVAEConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder = nn.Sequential(
            nn.Linear(config.descriptor_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.descriptor_dim),
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

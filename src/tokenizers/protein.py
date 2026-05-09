"""Protein pocket structure and sequence tokenization.

Provides:
- Pocket extraction from PDB files (precomputed + on-demand).
- Backbone spherical-from-pocket-centroid per-residue descriptor (65-D)
  with one-shot reconstruction.
- VQ-VAE for structure tokenization (via TransformerVQVAE).
- Simple amino acid sequence tokenizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from torch import Tensor, nn

from src.tokenizers.codebook import EMACodebook
from src.tokenizers.descriptor_schema import (
    K_NEIGHBORS,
    PROTEIN_AA_TO_IDX,
    PROTEIN_AA_VOCAB,
    PROTEIN_AA_X_IDX,
    PROTEIN_DESCRIPTOR_DIM,
    PROTEIN_LAYOUT,
    fields_by_name,
)
from src.tokenizers.geometry import (
    cartesian_to_spherical,
    spherical_to_cartesian,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.config import PocketExtractionConfig, ProteinVQVAEConfig

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


# ---------------------------------------------------------------------------
# Precomputed pocket extraction (for batch processing)
# ---------------------------------------------------------------------------


class PrecomputedResidues:
    """Precomputed residue data from a PDB file for fast pocket extraction.

    Parse the PDB once with :func:`precompute_pocket_candidates`, then call
    :func:`extract_pocket_from_candidates` for each ligand.
    """

    __slots__ = (
        "backbone_coords",
        "ca_coords",
        "chain_ids",
        "residue_indices",
        "residue_names",
    )

    def __init__(
        self,
        ca_coords: np.ndarray,
        backbone_coords: np.ndarray,
        chain_ids: list[str],
        residue_indices: list[int],
        residue_names: list[str],
    ) -> None:
        self.ca_coords = ca_coords
        self.backbone_coords = backbone_coords
        self.chain_ids = chain_ids
        self.residue_indices = residue_indices
        self.residue_names = residue_names


def precompute_pocket_candidates(pdb_path: str | Path) -> PrecomputedResidues:
    """Parse a PDB file and return precomputed residue data."""
    from Bio.PDB import PDBParser  # noqa: PLC0415

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", str(pdb_path))
    model = structure[0]

    ca_list: list[np.ndarray] = []
    bb_list: list[np.ndarray] = []
    chain_ids: list[str] = []
    residue_indices: list[int] = []
    residue_names: list[str] = []

    for chain in model:
        for residue in chain:
            resname = residue.get_resname()
            if resname not in AA_3TO1:
                continue
            if not all(atom in residue for atom in BACKBONE_ATOMS):
                continue
            ca_list.append(residue["CA"].get_vector().get_array())
            bb_list.append(
                np.array(
                    [residue[a].get_vector().get_array() for a in BACKBONE_ATOMS],
                    dtype=np.float32,
                )
            )
            chain_ids.append(chain.id)
            residue_indices.append(residue.get_id()[1])
            residue_names.append(resname)

    return PrecomputedResidues(
        ca_coords=np.array(ca_list, dtype=np.float32),
        backbone_coords=(
            np.array(bb_list, dtype=np.float32)
            if bb_list
            else np.empty((0, 3, 3), dtype=np.float32)
        ),
        chain_ids=chain_ids,
        residue_indices=residue_indices,
        residue_names=residue_names,
    )


def extract_pocket_from_candidates(
    precomputed: PrecomputedResidues,
    ligand_coords: np.ndarray,
    config: PocketExtractionConfig,
) -> tuple[np.ndarray, str, list[tuple[str, int]]] | None:
    """Extract pocket residues from precomputed data."""
    if len(precomputed.ca_coords) == 0:
        return None

    diff = precomputed.ca_coords[:, None, :] - ligand_coords[None, :, :]
    min_dists = np.linalg.norm(diff, axis=2).min(axis=1)

    within = np.where(min_dists <= config.distance_cutoff)[0]
    if len(within) == 0:
        return None

    order = np.argsort(min_dists[within])
    selected = within[order][: config.max_residues]

    sort_keys = [
        (precomputed.chain_ids[i], precomputed.residue_indices[i]) for i in selected
    ]
    final_order = sorted(range(len(selected)), key=lambda k: sort_keys[k])
    selected = selected[final_order].tolist()

    backbone_coords = precomputed.backbone_coords[selected]
    pocket_seq = "".join(AA_3TO1[precomputed.residue_names[i]] for i in selected)
    residue_ids = [
        (precomputed.chain_ids[i], precomputed.residue_indices[i]) for i in selected
    ]

    return backbone_coords, pocket_seq, residue_ids


def extract_pocket(
    pdb_path: str | Path,
    ligand_coords: np.ndarray,
    config: PocketExtractionConfig,
) -> tuple[np.ndarray, str, list[tuple[str, int]]] | None:
    """Extract pocket residues near the ligand from a PDB file."""
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
            if not all(atom in residue for atom in BACKBONE_ATOMS):
                continue
            ca_coord = residue["CA"].get_vector().get_array()
            min_dist = np.min(np.linalg.norm(ligand_coords - ca_coord, axis=1))
            if min_dist <= config.distance_cutoff:
                residues_with_dist.append((min_dist, residue))

    if not residues_with_dist:
        return None

    residues_with_dist.sort(key=lambda x: x[0])
    selected = [r for _, r in residues_with_dist[: config.max_residues]]
    selected.sort(key=lambda r: (r.get_parent().id, r.get_id()[1]))

    backbone_coords = []
    pocket_seq = []
    residue_ids: list[tuple[str, int]] = []
    for residue in selected:
        coords = [residue[atom].get_vector().get_array() for atom in BACKBONE_ATOMS]
        backbone_coords.append(coords)
        pocket_seq.append(AA_3TO1[residue.get_resname()])
        residue_ids.append((residue.get_parent().id, residue.get_id()[1]))

    return (
        np.array(backbone_coords, dtype=np.float32),
        "".join(pocket_seq),
        residue_ids,
    )


def extract_full_sequence(pdb_path: str | Path) -> str:
    """Extract full amino acid sequence from a PDB file."""
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
# Canonical pocket frame (PCA on CA coords with sign disambiguation)
# ---------------------------------------------------------------------------


def _compute_canonical_frame(
    ca_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic canonical frame from CA coordinates via PCA."""
    centroid = ca_coords.mean(axis=0).astype(np.float64)
    centered = (ca_coords - centroid).astype(np.float64)

    if len(ca_coords) < 2:  # noqa: PLR2004
        return centroid, np.eye(3, dtype=np.float64)

    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    if vt.shape[0] < 3:  # noqa: PLR2004
        vt_full = np.eye(3, dtype=np.float64)
        vt_full[: vt.shape[0]] = vt
        if vt.shape[0] == 1:
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

    for i in range(3):
        proj = centered @ vt[i]
        max_idx = int(np.argmax(np.abs(proj)))
        if proj[max_idx] < 0:
            vt[i] *= -1

    if np.linalg.det(vt) < 0:
        vt[2] *= -1

    return centroid, vt.astype(np.float64)


# ---------------------------------------------------------------------------
# Backbone spherical descriptor
# ---------------------------------------------------------------------------


def _knn_residue_offsets_and_aa(
    canonical_backbone: np.ndarray,  # (L, 3, 3): N, CA, C in canonical frame
    aa_indices: np.ndarray,  # (L,)
    k: int = K_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-residue KNN over CA distances. Offsets carry all 3 backbone atoms.

    For each residue i, the k nearest residues j (by CA distance) contribute
    spherical offsets ``(r, θ, sin φ, cos φ)`` for each of (N_j - CA_i,
    CA_j - CA_i, C_j - CA_i). Output has shape ``(L, k * 12)`` for offsets
    and ``(L, k)`` for AA indices.
    """
    n = canonical_backbone.shape[0]
    offsets = np.zeros((n, k * 12), dtype=np.float32)
    nbr_aa = np.full((n, k), PROTEIN_AA_X_IDX, dtype=np.int64)

    if n <= 1:
        return offsets, nbr_aa

    ca = canonical_backbone[:, 1, :]
    diff = ca[:, None, :] - ca[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)

    take = min(k, n - 1)
    nbr_idx = np.argpartition(dist, take - 1, axis=1)[:, :take]
    for i in range(n):
        order = np.argsort(dist[i, nbr_idx[i]])
        nbr_idx[i] = nbr_idx[i, order]

    for i in range(n):
        ca_i = canonical_backbone[i, 1]
        for slot, j in enumerate(nbr_idx[i]):
            for atom_local in range(3):
                delta = canonical_backbone[j, atom_local] - ca_i
                r, theta, sphi, cphi = cartesian_to_spherical(delta)
                base = slot * 12 + atom_local * 4
                offsets[i, base : base + 4] = (r, theta, sphi, cphi)
            nbr_aa[i, slot] = aa_indices[j]
    return offsets, nbr_aa


class BackboneSphericalDescriptor:
    """Per-residue backbone descriptor: spherical from pocket centroid + AA + KNN.

    Each residue gets a 65-D row laid out by :data:`PROTEIN_LAYOUT`:

    - 12 continuous: (N, CA, C) x (r, θ, sin φ, cos φ) from pocket centroid in
      the canonical frame. No segment / NeRF chain — every residue is encoded
      independently and decoded with one ``spherical → Cartesian → frame`` step.
    - 1 categorical: amino acid index (0..20, ``X`` is the catch-all).
    - 48 continuous + 4 categorical: KNN encoder hint features over the
      pocket's other residues (CA-based, K=4 by default).
    """

    DESCRIPTOR_DIM = PROTEIN_DESCRIPTOR_DIM

    def compute(
        self,
        backbone_coords: np.ndarray,  # (L, 3, 3) global frame: N, CA, C
        residue_ids: list[tuple[str, int]],
        pocket_frame: tuple[np.ndarray, np.ndarray] | None = None,
        residue_names_one_letter: list[str] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Compute descriptors for all residues.

        Args:
            backbone_coords: ``(L, 3, 3)`` (N, CA, C) per residue in global frame.
            residue_ids: ``(chain_id, residue_index)`` per residue.
            pocket_frame: ``(centroid, rotation)`` — if ``None`` the frame is
                computed from CA coordinates.
            residue_names_one_letter: optional per-residue AA letter; if
                missing, every residue is set to ``X``. Pass
                ``list(pocket_seq)`` from :func:`extract_pocket_from_candidates`.

        Returns:
            descriptors: ``(L, 65)`` float32.
            metadata: dict with ``centroid``, ``rotation``, ``residue_ids``.
        """
        n = len(backbone_coords)
        bb = backbone_coords.astype(np.float64)
        ca = bb[:, 1]

        if pocket_frame is None:
            centroid, rotation = _compute_canonical_frame(ca)
        else:
            centroid, rotation = pocket_frame
        centroid = centroid.astype(np.float64)
        rotation = rotation.astype(np.float64)

        canonical = np.zeros_like(bb)
        for i in range(n):
            for j in range(3):
                canonical[i, j] = (bb[i, j] - centroid) @ rotation.T

        # Spherical (r, θ, sin φ, cos φ) for each of (N, CA, C) per residue.
        coord = np.zeros((n, 12), dtype=np.float32)
        for i in range(n):
            for atom_local in range(3):
                r, theta, sphi, cphi = cartesian_to_spherical(canonical[i, atom_local])
                coord[i, atom_local * 4 : atom_local * 4 + 4] = (r, theta, sphi, cphi)

        # AA indices.
        aa = np.full(n, PROTEIN_AA_X_IDX, dtype=np.int64)
        if residue_names_one_letter is not None:
            for i, letter in enumerate(residue_names_one_letter[:n]):
                aa[i] = PROTEIN_AA_TO_IDX.get(letter, PROTEIN_AA_X_IDX)

        # KNN encoder hints.
        knn_offsets, knn_aa = _knn_residue_offsets_and_aa(canonical, aa)

        descriptor = np.zeros((n, PROTEIN_DESCRIPTOR_DIM), dtype=np.float32)
        f = fields_by_name(PROTEIN_LAYOUT)
        descriptor[:, f["coord"].start : f["coord"].end] = coord
        descriptor[:, f["aa"].start] = aa
        descriptor[:, f["knn_offsets"].start : f["knn_offsets"].end] = knn_offsets
        descriptor[:, f["knn_aa"].start : f["knn_aa"].end] = knn_aa

        metadata: dict[str, Any] = {
            "centroid": centroid,
            "rotation": rotation,
            "residue_ids": residue_ids,
        }
        return descriptor, metadata

    @staticmethod
    def descriptor_to_backbone_coords(
        descriptors: np.ndarray,
        metadata: dict[str, Any],
    ) -> np.ndarray:
        """Reconstruct global ``(L, 3, 3)`` backbone coords from descriptors."""
        centroid = np.asarray(metadata["centroid"], dtype=np.float64)
        rotation = np.asarray(metadata["rotation"], dtype=np.float64)

        n = descriptors.shape[0]
        if n == 0:
            return np.zeros((0, 3, 3), dtype=np.float32)

        f = fields_by_name(PROTEIN_LAYOUT)
        coord = descriptors[:, f["coord"].start : f["coord"].end].astype(np.float64)

        canonical = np.zeros((n, 3, 3), dtype=np.float64)
        for i in range(n):
            for atom_local in range(3):
                base = atom_local * 4
                r = float(coord[i, base])
                theta = float(coord[i, base + 1])
                sphi = float(coord[i, base + 2])
                cphi = float(coord[i, base + 3])
                canonical[i, atom_local] = spherical_to_cartesian(r, theta, sphi, cphi)

        backbone = np.zeros_like(canonical)
        for i in range(n):
            for j in range(3):
                backbone[i, j] = canonical[i, j] @ rotation + centroid

        return backbone.astype(np.float32)


# ---------------------------------------------------------------------------
# Legacy VQ-VAE (kept for backward compatibility with old checkpoints / tests)
# ---------------------------------------------------------------------------


class ProteinStructureVQVAE(nn.Module):
    """VQ-VAE for protein backbone structure tokenization.

    .. deprecated::
        Use :class:`~src.tokenizers.vqvae.TransformerVQVAE` for new code.
    """

    def __init__(self, config: ProteinVQVAEConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder = nn.Sequential(
            nn.Linear(config.descriptor_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
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
        z = self.encoder(x)
        quantized, indices, commitment_loss, _ = self.codebook(z)
        reconstructed = self.decoder(quantized)
        reconstruction_loss = (x - reconstructed).pow(2).mean()
        return {
            "reconstructed": reconstructed,
            "indices": indices,
            "commitment_loss": commitment_loss,
            "reconstruction_loss": reconstruction_loss,
        }

    def encode(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        _, indices, _, _ = self.codebook(z)
        return indices

    def decode(self, indices: Tensor) -> Tensor:
        quantized = self.codebook.lookup(indices)
        return self.decoder(quantized)


class ProteinSequenceTokenizer:
    """Simple amino acid sequence tokenizer (no learning required)."""

    VOCAB: ClassVar[list[str]] = list(PROTEIN_AA_VOCAB)

    def __init__(self) -> None:
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.VOCAB)}

    def encode(self, sequence: str) -> list[str]:
        return [c if c in self.aa_to_idx else "X" for c in sequence]

    @property
    def vocab_size(self) -> int:
        return len(self.VOCAB)

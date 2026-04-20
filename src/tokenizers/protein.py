"""Protein pocket structure and sequence tokenization.

Provides:
- Pocket extraction from PDB files
- Backbone Z-matrix per-residue descriptor (12D, invertible via NeRF)
- PCA canonical-frame per-residue descriptor (9D, legacy)
- VQ-VAE for structure tokenization (via TransformerVQVAE)
- Simple amino acid sequence tokenizer
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from torch import Tensor, nn

from src.tokenizers.codebook import EMACodebook
from src.tokenizers.geometry import (
    _EPS,
    bond_angle,
    canonical_virtual_ref,
    cartesian_to_spherical,
    dihedral_angle,
    place_atom,
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
        self.ca_coords = ca_coords  # (N_residues, 3)
        self.backbone_coords = backbone_coords  # (N_residues, 3, 3)
        self.chain_ids = chain_ids
        self.residue_indices = residue_indices
        self.residue_names = residue_names


def precompute_pocket_candidates(pdb_path: str | Path) -> PrecomputedResidues:
    """Parse a PDB file and return precomputed residue data.

    This avoids re-parsing the PDB for every ligand pose when multiple
    poses share the same receptor.
    """
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
    """Extract pocket residues from precomputed data.

    Same return format as :func:`extract_pocket` but avoids re-parsing
    the PDB file.
    """
    if len(precomputed.ca_coords) == 0:
        return None

    # Vectorised min-distance: (N_res, N_lig) -> (N_res,)
    diff = precomputed.ca_coords[:, None, :] - ligand_coords[None, :, :]
    min_dists = np.linalg.norm(diff, axis=2).min(axis=1)

    within = np.where(min_dists <= config.distance_cutoff)[0]
    if len(within) == 0:
        return None

    # Sort by distance, truncate
    order = np.argsort(min_dists[within])
    selected = within[order][: config.max_residues]

    # Sort by (chain_id, residue_index)
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
    """Extract pocket residues near the ligand from a PDB file.

    Args:
        pdb_path: Path to the receptor PDB file.
        ligand_coords: Ligand heavy-atom coordinates, shape ``(N_lig, 3)``.
        config: Pocket extraction parameters.

    Returns:
        Tuple of ``(backbone_coords, pocket_sequence, residue_ids)`` where
        *backbone_coords* has shape ``(L, 3, 3)`` for (N, CA, C),
        *pocket_sequence* is a string of 1-letter amino acid codes, and
        *residue_ids* is a list of ``(chain_id, residue_index)`` tuples
        for segment boundary detection.  Returns ``None`` if extraction
        fails.
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
# PCA canonical frame helpers (kept for BackboneZMatrixDescriptor anchoring)
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
# Segment detection
# ---------------------------------------------------------------------------


def _detect_segments(
    residue_ids: list[tuple[str, int]],
) -> list[tuple[int, int]]:
    """Detect contiguous residue segments in the pocket.

    Two residues are contiguous if they share the same chain and their
    residue indices differ by exactly 1.

    Args:
        residue_ids: List of ``(chain_id, residue_index)`` sorted by
            ``(chain_id, residue_index)``.

    Returns:
        List of ``(start, end)`` index pairs (half-open intervals) into
        *residue_ids* for each contiguous segment.
    """
    if not residue_ids:
        return []

    segments: list[tuple[int, int]] = []
    seg_start = 0

    for i in range(1, len(residue_ids)):
        prev_chain, prev_idx = residue_ids[i - 1]
        curr_chain, curr_idx = residue_ids[i]
        if curr_chain != prev_chain or curr_idx != prev_idx + 1:
            segments.append((seg_start, i))
            seg_start = i

    segments.append((seg_start, len(residue_ids)))
    return segments


# ---------------------------------------------------------------------------
# Backbone Z-matrix descriptor
# ---------------------------------------------------------------------------


class BackboneZMatrixDescriptor:
    """Backbone Z-matrix per-residue descriptor with NeRF inverse.

    Each residue receives a **12-D** descriptor by concatenating 4-D
    Z-matrix entries for each backbone atom (N, CA, C):

    *   ``(bond_length, bond_angle, sin_torsion, cos_torsion)``

    For the first residue of each contiguous segment the descriptor
    encodes pocket-frame-anchored coordinates (analogous to the ligand
    tokenizer's pocket-anchored mode).

    For subsequent residues within a segment the descriptor uses
    standard internal coordinates referencing the previous residue's
    backbone atoms.
    """

    DESCRIPTOR_DIM = 12

    def compute(
        self,
        backbone_coords: np.ndarray,
        residue_ids: list[tuple[str, int]],
        pocket_frame: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Compute 12-D backbone Z-matrix descriptors for all residues.

        Args:
            backbone_coords: Shape ``(L, 3, 3)`` — (N, CA, C) per residue.
            residue_ids: ``(chain_id, residue_index)`` per residue,
                sorted by ``(chain_id, residue_index)``.
            pocket_frame: ``(centroid, rotation)`` from PCA canonical
                frame.  When ``None``, a frame is computed internally.

        Returns:
            descriptors: ``(L, 12)`` float32 array.
            metadata: dict with keys ``centroid``, ``rotation``,
                ``segments``, ``residue_ids``.
        """
        num_residues = len(backbone_coords)
        bb = backbone_coords.astype(np.float64)

        # Build canonical frame if not provided
        ca_coords = bb[:, 1]  # (L, 3) — CA positions
        if pocket_frame is not None:
            centroid, rotation = pocket_frame
        else:
            centroid, rotation = _compute_canonical_frame(ca_coords)
        centroid = centroid.astype(np.float64)
        rotation = rotation.astype(np.float64)

        # Transform to canonical frame
        wc = np.zeros_like(bb)
        for i in range(num_residues):
            for j in range(3):
                wc[i, j] = (bb[i, j] - centroid) @ rotation.T

        segments = _detect_segments(residue_ids)

        descriptors = np.zeros((num_residues, 12), dtype=np.float64)

        for seg_start, seg_end in segments:
            for k, idx in enumerate(range(seg_start, seg_end)):
                n_pos = wc[idx, 0]
                ca_pos = wc[idx, 1]
                c_pos = wc[idx, 2]

                if k == 0:
                    # --- Segment start: pocket-frame anchored ---
                    self._encode_segment_start(
                        descriptors,
                        idx,
                        n_pos,
                        ca_pos,
                        c_pos,
                    )
                else:
                    # --- Continuation: Z-matrix referencing prev ---
                    prev = idx - 1
                    prev_n = wc[prev, 0]
                    prev_ca = wc[prev, 1]
                    prev_c = wc[prev, 2]
                    self._encode_continuation(
                        descriptors,
                        idx,
                        n_pos,
                        ca_pos,
                        c_pos,
                        prev_n,
                        prev_ca,
                        prev_c,
                    )

        metadata: dict[str, Any] = {
            "centroid": centroid,
            "rotation": rotation,
            "segments": segments,
            "residue_ids": residue_ids,
        }
        return descriptors.astype(np.float32), metadata

    @staticmethod
    def _encode_segment_start(
        descriptors: np.ndarray,
        idx: int,
        n_pos: np.ndarray,
        ca_pos: np.ndarray,
        c_pos: np.ndarray,
    ) -> None:
        """Encode the first residue of a segment (pocket-frame anchored)."""
        # N atom: spherical coords relative to origin (= pocket centroid)
        r, theta, sin_phi, cos_phi = cartesian_to_spherical(n_pos)
        descriptors[idx, 0:4] = [r, theta, sin_phi, cos_phi]

        # CA atom: bond length + direction from N in canonical frame
        d_n_ca = float(np.linalg.norm(ca_pos - n_pos))
        if d_n_ca >= _EPS:
            delta_hat = (ca_pos - n_pos) / d_n_ca
            _, th, sp, cp = cartesian_to_spherical(delta_hat)
        else:
            th, sp, cp = 0.0, 0.0, 1.0
        descriptors[idx, 4:8] = [d_n_ca, th, sp, cp]

        # C atom: bond length, bond angle, torsion w.r.t. canonical virtual ref
        d_ca_c = float(np.linalg.norm(c_pos - ca_pos))
        theta_nac = bond_angle(n_pos, ca_pos, c_pos)
        virtual = canonical_virtual_ref(n_pos, ca_pos)
        tau = dihedral_angle(virtual, n_pos, ca_pos, c_pos)
        descriptors[idx, 8:12] = [d_ca_c, theta_nac, np.sin(tau), np.cos(tau)]

    @staticmethod
    def _encode_continuation(  # noqa: PLR0913
        descriptors: np.ndarray,
        idx: int,
        n_pos: np.ndarray,
        ca_pos: np.ndarray,
        c_pos: np.ndarray,
        prev_n: np.ndarray,
        prev_ca: np.ndarray,
        prev_c: np.ndarray,
    ) -> None:
        """Encode a non-start residue using Z-matrix references."""
        # N(i): bond C(i-1)->N(i), angle CA(i-1)-C(i-1)-N(i),
        #        dihedral N(i-1)-CA(i-1)-C(i-1)-N(i) (= psi_{i-1})
        d_c_n = float(np.linalg.norm(n_pos - prev_c))
        angle_ca_c_n = bond_angle(prev_ca, prev_c, n_pos)
        psi_prev = dihedral_angle(prev_n, prev_ca, prev_c, n_pos)
        descriptors[idx, 0:4] = [
            d_c_n,
            angle_ca_c_n,
            np.sin(psi_prev),
            np.cos(psi_prev),
        ]

        # CA(i): bond N(i)->CA(i), angle C(i-1)-N(i)-CA(i),
        #         dihedral CA(i-1)-C(i-1)-N(i)-CA(i) (= omega_i)
        d_n_ca = float(np.linalg.norm(ca_pos - n_pos))
        angle_c_n_ca = bond_angle(prev_c, n_pos, ca_pos)
        omega = dihedral_angle(prev_ca, prev_c, n_pos, ca_pos)
        descriptors[idx, 4:8] = [d_n_ca, angle_c_n_ca, np.sin(omega), np.cos(omega)]

        # C(i): bond CA(i)->C(i), angle N(i)-CA(i)-C(i),
        #        dihedral C(i-1)-N(i)-CA(i)-C(i) (= phi_i)
        d_ca_c = float(np.linalg.norm(c_pos - ca_pos))
        angle_n_ca_c = bond_angle(n_pos, ca_pos, c_pos)
        phi = dihedral_angle(prev_c, n_pos, ca_pos, c_pos)
        descriptors[idx, 8:12] = [d_ca_c, angle_n_ca_c, np.sin(phi), np.cos(phi)]

    # ---- inverse transform ------------------------------------------------

    @staticmethod
    def descriptor_to_backbone_coords(
        descriptors: np.ndarray,
        metadata: dict[str, Any],
    ) -> np.ndarray:
        """Reconstruct backbone ``(N, CA, C)`` from descriptors + metadata.

        Uses NeRF placement for continuation residues and
        pocket-frame-anchored decoding for segment starts.

        Args:
            descriptors: ``(L, 12)`` float32/64 array.
            metadata: As returned by :meth:`compute`.

        Returns:
            backbone_coords: ``(L, 3, 3)`` float32 array in global frame.
        """
        centroid: np.ndarray = metadata["centroid"].astype(np.float64)
        rotation: np.ndarray = metadata["rotation"].astype(np.float64)
        segments: list[tuple[int, int]] = metadata["segments"]

        num_residues = len(descriptors)
        desc = descriptors.astype(np.float64)
        # Reconstruct in canonical frame first
        wc = np.zeros((num_residues, 3, 3), dtype=np.float64)

        for seg_start, seg_end in segments:
            for k, idx in enumerate(range(seg_start, seg_end)):
                if k == 0:
                    _decode_segment_start(wc, desc, idx)
                else:
                    _decode_continuation(wc, desc, idx)

        # Transform from canonical frame to global
        backbone = np.zeros((num_residues, 3, 3), dtype=np.float64)
        for i in range(num_residues):
            for j in range(3):
                backbone[i, j] = wc[i, j] @ rotation + centroid

        return backbone.astype(np.float32)


def _decode_segment_start(
    wc: np.ndarray,
    desc: np.ndarray,
    idx: int,
) -> None:
    """Decode a segment-start residue from pocket-anchored encoding."""
    # N atom: spherical coords
    r = float(desc[idx, 0])
    theta = float(desc[idx, 1])
    sin_phi = float(desc[idx, 2])
    cos_phi = float(desc[idx, 3])
    wc[idx, 0] = spherical_to_cartesian(r, theta, sin_phi, cos_phi)

    # CA atom: bond length + direction from N
    d_n_ca = float(desc[idx, 4])
    th = float(desc[idx, 5])
    sp = float(desc[idx, 6])
    cp = float(desc[idx, 7])
    direction = spherical_to_cartesian(1.0, th, sp, cp)
    wc[idx, 1] = wc[idx, 0] + d_n_ca * direction

    # C atom: bond length, bond angle, torsion w.r.t. canonical virtual ref
    d_ca_c = float(desc[idx, 8])
    theta_nac = float(desc[idx, 9])
    sin_tau = float(desc[idx, 10])
    cos_tau = float(desc[idx, 11])
    tau = float(np.arctan2(sin_tau, cos_tau))
    virtual = canonical_virtual_ref(wc[idx, 0], wc[idx, 1])
    wc[idx, 2] = place_atom(virtual, wc[idx, 0], wc[idx, 1], d_ca_c, theta_nac, tau)


def _decode_continuation(
    wc: np.ndarray,
    desc: np.ndarray,
    idx: int,
) -> None:
    """Decode a continuation residue using NeRF placement."""
    prev = idx - 1
    prev_n = wc[prev, 0]
    prev_ca = wc[prev, 1]
    prev_c = wc[prev, 2]

    # N(i): placed from N(i-1), CA(i-1), C(i-1)
    d_c_n = float(desc[idx, 0])
    angle_ca_c_n = float(desc[idx, 1])
    psi_prev = float(np.arctan2(desc[idx, 2], desc[idx, 3]))
    wc[idx, 0] = place_atom(prev_n, prev_ca, prev_c, d_c_n, angle_ca_c_n, psi_prev)

    n_pos = wc[idx, 0]

    # CA(i): placed from CA(i-1), C(i-1), N(i)
    d_n_ca = float(desc[idx, 4])
    angle_c_n_ca = float(desc[idx, 5])
    omega = float(np.arctan2(desc[idx, 6], desc[idx, 7]))
    wc[idx, 1] = place_atom(prev_ca, prev_c, n_pos, d_n_ca, angle_c_n_ca, omega)

    ca_pos = wc[idx, 1]

    # C(i): placed from C(i-1), N(i), CA(i)
    d_ca_c = float(desc[idx, 8])
    angle_n_ca_c = float(desc[idx, 9])
    phi = float(np.arctan2(desc[idx, 10], desc[idx, 11]))
    wc[idx, 2] = place_atom(prev_c, n_pos, ca_pos, d_ca_c, angle_n_ca_c, phi)


# ---------------------------------------------------------------------------
# Legacy PCA descriptor (kept for backward compatibility)
# ---------------------------------------------------------------------------


class PocketDescriptor:
    """PCA canonical-frame per-residue descriptor with exact inverse.

    Each residue receives a **9-D** descriptor:

    *   CA position in canonical frame (3D)
    *   N-CA offset in canonical frame (3D)
    *   C-CA offset in canonical frame (3D)

    The canonical frame is derived from PCA of the pocket CA coordinates with
    sign disambiguation, ensuring SE(3)-invariant descriptors.

    .. deprecated::
        Use :class:`BackboneZMatrixDescriptor` for new code.
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
# Legacy VQ-VAE (kept for backward compatibility with old checkpoints)
# ---------------------------------------------------------------------------


class ProteinStructureVQVAE(nn.Module):
    """VQ-VAE for protein backbone structure tokenization.

    Encodes per-residue descriptors into discrete codes.

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
        """Forward pass: encode, quantize, decode.

        Args:
            x: Descriptors of shape ``(B, descriptor_dim)``.

        Returns:
            Dict with keys: reconstructed, indices, commitment_loss,
            reconstruction_loss.
        """
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
        """Encode descriptors to codebook indices."""
        z = self.encoder(x)
        _, indices, _, _ = self.codebook(z)
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

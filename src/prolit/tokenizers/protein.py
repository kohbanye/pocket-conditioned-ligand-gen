"""Protein pocket structure and sequence tokenization.

Provides:
- Pocket extraction from PDB files (precomputed + on-demand).
- Backbone spherical-from-pocket-centroid per-residue descriptor (65-D)
  with one-shot reconstruction.
- Simple amino acid sequence tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from Bio.PDB.Model import Model

    from prolit.config import PocketExtractionConfig

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


# ---------------------------------------------------------------------------
# All-atom pocket extraction (every heavy atom of the pocket residues)
# ---------------------------------------------------------------------------


def _atom_element(atom: object) -> str:
    """Best-effort element symbol for a BioPython atom (empty element column)."""
    elem = (getattr(atom, "element", "") or "").strip()
    if elem:
        return elem.capitalize()
    name = atom.get_name().strip()  # type: ignore[attr-defined]
    for ch in name:
        if ch.isalpha():
            return ch.upper()
    return ""


@dataclass
class PrecomputedPocketAtoms:
    """Per-receptor heavy-atom data for fast all-atom pocket extraction.

    Parse once with :func:`precompute_pocket_atom_candidates`, then call
    :func:`extract_pocket_atoms_from_candidates` per ligand pose. Heavy atoms
    only (hydrogens dropped); ``residue_atoms[i]`` lists ``(name, element,
    coord)`` for residue ``i``.
    """

    ca_coords: np.ndarray  # (R, 3)
    chain_ids: list[str]
    residue_indices: list[int]
    residue_names: list[str]  # 3-letter
    residue_atoms: list[list[tuple[str, str, np.ndarray]]]


@dataclass
class PocketAtomData:
    """Flattened heavy-atom view of the selected pocket residues.

    ``ca_coords`` are the selected residues' CA (used to build the shared
    canonical frame); the ``atom_*`` arrays are one row per heavy atom.
    """

    ca_coords: np.ndarray  # (L, 3) selected residues' CA
    atom_coords: np.ndarray  # (M, 3)
    atom_elements: list[str]  # (M,)
    atom_names: list[str]  # (M,)
    atom_aa: list[str]  # (M,) one-letter residue type
    atom_chain: list[str]  # (M,)
    atom_resseq: list[int]  # (M,)
    residue_ids: list[tuple[str, int]]  # (L,)
    pocket_seq: str = field(default="")


def precompute_pocket_atom_candidates(
    pdb_path: str | Path,
) -> PrecomputedPocketAtoms:
    """Parse a PDB file and return per-residue heavy-atom data (standard AAs)."""
    from Bio.PDB import PDBParser  # noqa: PLC0415

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", str(pdb_path))
    return _precompute_pocket_atoms_from_model(structure[0])


def precompute_pocket_atom_candidates_from_text(
    pdb_text: str,
) -> PrecomputedPocketAtoms:
    """Same as :func:`precompute_pocket_atom_candidates` but from PDB text.

    Used to stream receptor structures out of zip/tar archives without writing
    them to disk (inode-safe).
    """
    from io import StringIO  # noqa: PLC0415

    from Bio.PDB import PDBParser  # noqa: PLC0415

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", StringIO(pdb_text))
    return _precompute_pocket_atoms_from_model(structure[0])


def _precompute_pocket_atoms_from_model(model: Model) -> PrecomputedPocketAtoms:
    ca_list: list[np.ndarray] = []
    chain_ids: list[str] = []
    residue_indices: list[int] = []
    residue_names: list[str] = []
    residue_atoms: list[list[tuple[str, str, np.ndarray]]] = []

    for chain in model:
        for residue in chain:
            resname = residue.get_resname()
            if resname not in AA_3TO1 or "CA" not in residue:
                continue
            atoms: list[tuple[str, str, np.ndarray]] = []
            for atom in residue:
                elem = _atom_element(atom)
                if elem in ("H", "D", ""):
                    continue
                atoms.append(
                    (
                        atom.get_name().strip(),
                        elem,
                        atom.get_vector().get_array().astype(np.float32),
                    )
                )
            if not atoms:
                continue
            ca_list.append(residue["CA"].get_vector().get_array())
            chain_ids.append(chain.id)
            residue_indices.append(residue.get_id()[1])
            residue_names.append(resname)
            residue_atoms.append(atoms)

    return PrecomputedPocketAtoms(
        ca_coords=(
            np.array(ca_list, dtype=np.float32)
            if ca_list
            else np.empty((0, 3), dtype=np.float32)
        ),
        chain_ids=chain_ids,
        residue_indices=residue_indices,
        residue_names=residue_names,
        residue_atoms=residue_atoms,
    )


def extract_pocket_atoms_from_candidates(
    precomputed: PrecomputedPocketAtoms,
    ligand_coords: np.ndarray,
    config: PocketExtractionConfig,
) -> PocketAtomData | None:
    """Select pocket residues by CA distance and flatten their heavy atoms."""
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

    ca_coords = precomputed.ca_coords[selected]
    residue_ids = [
        (precomputed.chain_ids[i], precomputed.residue_indices[i]) for i in selected
    ]
    pocket_seq = "".join(AA_3TO1[precomputed.residue_names[i]] for i in selected)

    atom_coords: list[np.ndarray] = []
    atom_elements: list[str] = []
    atom_names: list[str] = []
    atom_aa: list[str] = []
    atom_chain: list[str] = []
    atom_resseq: list[int] = []
    for i in selected:
        aa1 = AA_3TO1[precomputed.residue_names[i]]
        chain = precomputed.chain_ids[i]
        resseq = precomputed.residue_indices[i]
        for name, elem, coord in precomputed.residue_atoms[i]:
            atom_coords.append(coord)
            atom_elements.append(elem)
            atom_names.append(name)
            atom_aa.append(aa1)
            atom_chain.append(chain)
            atom_resseq.append(resseq)

    return PocketAtomData(
        ca_coords=ca_coords.astype(np.float32),
        atom_coords=np.array(atom_coords, dtype=np.float32),
        atom_elements=atom_elements,
        atom_names=atom_names,
        atom_aa=atom_aa,
        atom_chain=atom_chain,
        atom_resseq=atom_resseq,
        residue_ids=residue_ids,
        pocket_seq=pocket_seq,
    )


# ---------------------------------------------------------------------------
# Canonical pocket frame (PCA on CA coords with sign disambiguation)
# ---------------------------------------------------------------------------


def compute_canonical_frame(
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



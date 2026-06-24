"""Unified all-atom descriptor for protein pocket atoms AND ligand atoms.

A single 33-D per-atom descriptor (layout :data:`ATOM_LAYOUT`) is used for both
domains so one VQ-VAE / one codebook can tokenize the whole complex. Every atom
row carries:

- pocket-anchored spherical coords ``(r, θ, sin φ, cos φ)`` (same canonical
  frame for protein and ligand),
- a ``source`` flag (protein / ligand) consumed by the encoder,
- ligand-parity chemistry features (element, charge, hybrid, aromatic, ring,
  numH) — for protein atoms these come from an RDKit parse of the receptor,
- protein-context features (residue type ``aa`` and backbone/side-chain flag
  ``bb_sc``); ligand atoms take the ``X`` / ``NA`` placeholder buckets,
- K=4 same-source nearest-neighbour spherical offsets + element indices.

The heavy lifting (RDKit feature extraction, KNN encoder hints) is reused from
:mod:`src.tokenizers.ligand`; this module only assembles the unified layout and
adds the protein-atom front-end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.tokenizers.descriptor_schema import (
    ATOM_DESCRIPTOR_DIM,
    ATOM_LAYOUT,
    BB_SC_BACKBONE_IDX,
    BB_SC_NA_IDX,
    BB_SC_SIDECHAIN_IDX,
    K_NEIGHBORS,
    LIGAND_CHARGE_TO_IDX,
    LIGAND_CHARGE_VOCAB,
    LIGAND_ELEMENT_TO_IDX,
    LIGAND_HYBRID_OTHER_IDX,
    LIGAND_NUMH_VOCAB,
    LIGAND_OTHER_IDX,
    LIGAND_RING_NONE_IDX,
    PROTEIN_AA_TO_IDX,
    PROTEIN_AA_X_IDX,
    PROTEIN_BACKBONE_ATOM_NAMES,
    SOURCE_LIGAND_IDX,
    SOURCE_PROTEIN_IDX,
    fields_by_name,
)
from src.tokenizers.geometry import (
    cartesian_to_spherical_np,
    spherical_to_cartesian_np,
)
from src.tokenizers.ligand import (
    _atom_features_from_mol,
    _build_rdkit_mol,
    _knn_offsets_and_elements,
)

if TYPE_CHECKING:
    from pathlib import Path

    from rdkit.Chem.rdchem import Atom, RingInfo

    from src.tokenizers.protein import PocketAtomData

# Default chemistry feature indices for atoms whose RDKit features are
# unavailable (neutral charge, unspecified hybridization, non-aromatic,
# not-in-ring, zero implicit H). Order matches ``_CHEM_FIELDS``.
_DEFAULT_CHEM = (
    LIGAND_CHARGE_TO_IDX[0],
    LIGAND_HYBRID_OTHER_IDX,
    0,
    LIGAND_RING_NONE_IDX,
    0,
)
_CHEM_FIELDS = ("charge", "hybrid", "aromatic", "ring", "numH")


# ---------------------------------------------------------------------------
# Shared assembly
# ---------------------------------------------------------------------------


def _assemble_atom_descriptor(  # noqa: PLR0913
    canonical: np.ndarray,  # (N, 3) canonical-frame coords
    source_idx: int,
    element_idx: np.ndarray,  # (N,)
    chem: dict[str, np.ndarray],  # charge/hybrid/aromatic/ring/numH -> (N,)
    aa_idx: np.ndarray,  # (N,)
    bb_sc_idx: np.ndarray,  # (N,)
) -> np.ndarray:
    """Fill an ``(N, ATOM_DESCRIPTOR_DIM)`` descriptor from per-atom features."""
    n = canonical.shape[0]
    desc = np.zeros((n, ATOM_DESCRIPTOR_DIM), dtype=np.float32)
    if n == 0:
        return desc
    f = fields_by_name(ATOM_LAYOUT)

    desc[:, f["coord"].start : f["coord"].end] = cartesian_to_spherical_np(
        canonical.astype(np.float64),
    ).astype(np.float32)
    desc[:, f["source"].start] = source_idx
    desc[:, f["element"].start] = element_idx
    for name in _CHEM_FIELDS:
        desc[:, f[name].start] = chem[name]
    desc[:, f["aa"].start] = aa_idx
    desc[:, f["bb_sc"].start] = bb_sc_idx

    knn_offsets, knn_elements = _knn_offsets_and_elements(canonical, element_idx)
    desc[:, f["knn_offsets"].start : f["knn_offsets"].end] = knn_offsets
    desc[:, f["knn_elements"].start : f["knn_elements"].end] = knn_elements
    return desc


def rotate_atom_descriptor(
    descriptor: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """Re-express an all-atom descriptor under an extra frame rotation ``R``.

    Generalises :func:`src.tokenizers.ligand.rotate_ligand_descriptor` to the
    unified layout: only the absolute ``coord`` block and each ``knn_offsets``
    block are orientation-dependent; every categorical slot (source / element /
    chemistry / aa / bb_sc / knn elements) is rotation-invariant and copied
    through. The same ``R`` must be applied to the protein AND ligand
    descriptors of a complex so the two stay in one frame.
    """
    if descriptor.shape[0] == 0:
        return descriptor.copy()

    out = descriptor.astype(np.float64, copy=True)
    f = fields_by_name(ATOM_LAYOUT)

    def _rotate_block(start: int) -> None:
        sph = out[:, start : start + 4]
        cart = spherical_to_cartesian_np(sph) @ rotation.T
        out[:, start : start + 4] = cartesian_to_spherical_np(cart)

    _rotate_block(f["coord"].start)
    knn = f["knn_offsets"]
    for k in range(knn.length // 4):
        _rotate_block(knn.start + 4 * k)

    return out.astype(np.float32)


def atom_descriptor_to_coords(
    descriptors: np.ndarray,
    metadata: dict[str, Any],
    pocket_frame: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Reconstruct global Cartesian coords from the ``coord`` block (N, 3)."""
    frame = pocket_frame or (metadata["centroid"], metadata["rotation"])
    centroid, rotation = frame
    n = descriptors.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    f = fields_by_name(ATOM_LAYOUT)
    coord = descriptors[:, f["coord"].start : f["coord"].end].astype(np.float64)
    canonical = spherical_to_cartesian_np(coord)
    return canonical @ rotation + centroid


# ---------------------------------------------------------------------------
# Ligand front-end
# ---------------------------------------------------------------------------


class LigandAtomDescriptor:
    """Unified-layout descriptor for ligand heavy atoms (source = ligand)."""

    DESCRIPTOR_DIM = ATOM_DESCRIPTOR_DIM

    def compute(
        self,
        atoms: list[tuple[str, float, float, float]],
        bonds: list[tuple[int, int, int]],
        pocket_frame: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, list[str], dict[str, Any]]:
        """Compute ``(N_heavy, 33)`` descriptors for the ligand's heavy atoms."""
        if pocket_frame is None:
            msg = "LigandAtomDescriptor.compute requires a pocket_frame"
            raise ValueError(msg)

        heavy_indices = [i for i, (e, *_) in enumerate(atoms) if e != "H"]
        centroid, rotation = pocket_frame
        if not heavy_indices:
            return (
                np.zeros((0, ATOM_DESCRIPTOR_DIM), dtype=np.float32),
                [],
                {"centroid": centroid, "rotation": rotation, "heavy_to_orig": []},
            )

        orig_to_heavy = {orig: h for h, orig in enumerate(heavy_indices)}
        heavy_atoms = [atoms[i] for i in heavy_indices]
        heavy_bonds = [
            (orig_to_heavy[a], orig_to_heavy[b], bt)
            for a, b, bt in bonds
            if a in orig_to_heavy and b in orig_to_heavy
        ]

        n = len(heavy_atoms)
        coords_global = np.array(
            [(a[1], a[2], a[3]) for a in heavy_atoms],
            dtype=np.float64,
        )
        elements_sym = [a[0] for a in heavy_atoms]
        canonical = (coords_global - centroid) @ rotation.T

        mol = _build_rdkit_mol(heavy_atoms, heavy_bonds)
        feats = _atom_features_from_mol(mol, n)

        descriptor = _assemble_atom_descriptor(
            canonical,
            SOURCE_LIGAND_IDX,
            feats["element"],
            {name: feats[name] for name in _CHEM_FIELDS},
            np.full(n, PROTEIN_AA_X_IDX, dtype=np.int64),
            np.full(n, BB_SC_NA_IDX, dtype=np.int64),
        )
        metadata: dict[str, Any] = {
            "centroid": centroid,
            "rotation": rotation,
            "heavy_to_orig": heavy_indices,
        }
        return descriptor, elements_sym, metadata


# ---------------------------------------------------------------------------
# Protein front-end (Full ligand-parity chemistry via an RDKit receptor parse)
# ---------------------------------------------------------------------------


def _chem_feature_indices(atom: Atom, ring_info: RingInfo) -> tuple[int, ...]:
    """Per-atom ``(charge, hybrid, aromatic, ring, numH)`` indices for an RDKit atom.

    Mirrors the per-atom logic of
    :func:`src.tokenizers.ligand._atom_features_from_mol` so protein and ligand
    atoms map to the same buckets.
    """
    from rdkit import Chem  # noqa: PLC0415

    hybrid_map = {
        Chem.HybridizationType.SP: 0,
        Chem.HybridizationType.SP2: 1,
        Chem.HybridizationType.SP3: 2,
    }
    idx = atom.GetIdx()

    c = max(
        min(atom.GetFormalCharge(), LIGAND_CHARGE_VOCAB[-1]), LIGAND_CHARGE_VOCAB[0]
    )
    charge = LIGAND_CHARGE_TO_IDX[c]

    is_arom = bool(atom.GetIsAromatic())
    aromatic = 1 if is_arom else 0
    hybrid = (
        3
        if is_arom
        else hybrid_map.get(atom.GetHybridization(), LIGAND_HYBRID_OTHER_IDX)
    )

    smallest = 0
    for size in (3, 4, 5, 6):
        if ring_info.IsAtomInRingOfSize(idx, size):
            smallest = size
            break
    else:
        if atom.IsInRing():
            smallest = 7
    if smallest == 0:
        ring = LIGAND_RING_NONE_IDX
    elif smallest <= 5:  # noqa: PLR2004
        ring = smallest - 3
    else:
        ring = 3

    nh = atom.GetTotalNumHs(includeNeighbors=False)
    numh = max(0, min(nh, LIGAND_NUMH_VOCAB[-1]))
    return charge, hybrid, aromatic, ring, numh


def precompute_receptor_atom_features(
    pdb_path: str | Path,
) -> dict[tuple[str, int, str], tuple[int, ...]]:
    """Parse a receptor PDB once with RDKit; map atoms to chemistry indices.

    Returns ``{(chain_id, residue_number, atom_name): (charge, hybrid,
    aromatic, ring, numH)}``. Missing keys at lookup time fall back to
    :data:`_DEFAULT_CHEM`. Returns an empty dict (so all atoms use defaults) if
    RDKit cannot parse / sanitise the receptor.
    """
    from rdkit import Chem  # noqa: PLC0415

    mol = Chem.MolFromPDBFile(str(pdb_path), sanitize=False, removeHs=True)
    return _receptor_atom_features_from_mol(mol)


def precompute_receptor_atom_features_from_text(
    pdb_text: str,
) -> dict[tuple[str, int, str], tuple[int, ...]]:
    """Same as :func:`precompute_receptor_atom_features` but from PDB text.

    Used to stream receptor structures out of zip/tar archives (inode-safe).
    """
    from rdkit import Chem  # noqa: PLC0415

    mol = Chem.MolFromPDBBlock(pdb_text, sanitize=False, removeHs=True)
    return _receptor_atom_features_from_mol(mol)


def _receptor_atom_features_from_mol(
    mol: object,
) -> dict[tuple[str, int, str], tuple[int, ...]]:
    from rdkit import Chem  # noqa: PLC0415

    if mol is None:
        return {}
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(mol)
        except Exception:  # noqa: BLE001
            return {}

    ring_info = mol.GetRingInfo()
    out: dict[tuple[str, int, str], tuple[int, ...]] = {}
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        key = (
            info.GetChainId().strip(),
            int(info.GetResidueNumber()),
            info.GetName().strip(),
        )
        out[key] = _chem_feature_indices(atom, ring_info)
    return out


class ProteinAtomDescriptor:
    """Unified-layout descriptor for protein pocket heavy atoms (source = protein)."""

    DESCRIPTOR_DIM = ATOM_DESCRIPTOR_DIM

    def compute(
        self,
        pocket_atoms: PocketAtomData,
        receptor_feats: dict[tuple[str, int, str], tuple[int, ...]],
        pocket_frame: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Compute ``(M, 33)`` descriptors for the pocket's heavy atoms.

        Args:
            pocket_atoms: extracted heavy-atom data
                (:class:`PocketAtomData`).
            receptor_feats: lookup from
                :func:`precompute_receptor_atom_features`.
            pocket_frame: ``(centroid, rotation)`` shared with the ligand.
        """
        centroid, rotation = pocket_frame
        coords = pocket_atoms.atom_coords.astype(np.float64)
        m = coords.shape[0]
        if m == 0:
            return (
                np.zeros((0, ATOM_DESCRIPTOR_DIM), dtype=np.float32),
                {"centroid": centroid, "rotation": rotation, "residue_ids": []},
            )
        canonical = (coords - centroid) @ rotation.T

        element_idx = np.array(
            [
                LIGAND_ELEMENT_TO_IDX.get(e, LIGAND_OTHER_IDX)
                for e in pocket_atoms.atom_elements
            ],
            dtype=np.int64,
        )
        aa_idx = np.array(
            [PROTEIN_AA_TO_IDX.get(a, PROTEIN_AA_X_IDX) for a in pocket_atoms.atom_aa],
            dtype=np.int64,
        )
        bb_sc_idx = np.array(
            [
                BB_SC_BACKBONE_IDX
                if name in PROTEIN_BACKBONE_ATOM_NAMES
                else BB_SC_SIDECHAIN_IDX
                for name in pocket_atoms.atom_names
            ],
            dtype=np.int64,
        )

        chem_rows = [
            receptor_feats.get(
                (chain, resseq, name),
                _DEFAULT_CHEM,
            )
            for chain, resseq, name in zip(
                pocket_atoms.atom_chain,
                pocket_atoms.atom_resseq,
                pocket_atoms.atom_names,
                strict=True,
            )
        ]
        chem_arr = np.array(chem_rows, dtype=np.int64)  # (M, 5)
        chem = {name: chem_arr[:, i] for i, name in enumerate(_CHEM_FIELDS)}

        descriptor = _assemble_atom_descriptor(
            canonical,
            SOURCE_PROTEIN_IDX,
            element_idx,
            chem,
            aa_idx,
            bb_sc_idx,
        )
        metadata: dict[str, Any] = {
            "centroid": centroid,
            "rotation": rotation,
            "residue_ids": pocket_atoms.residue_ids,
        }
        return descriptor, metadata


__all__ = [
    "K_NEIGHBORS",
    "LigandAtomDescriptor",
    "ProteinAtomDescriptor",
    "atom_descriptor_to_coords",
    "precompute_receptor_atom_features",
    "rotate_atom_descriptor",
]

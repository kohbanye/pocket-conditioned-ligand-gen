"""Ligand 3D structure tokenization with spherical coords + atom features.

Each heavy atom gets a single 30-D descriptor row whose layout is defined in
:mod:`src.tokenizers.descriptor_schema`. The descriptor combines:

- Pocket-anchored spherical coords ``(r, θ, sin φ, cos φ)`` from the pocket
  centroid in the canonical frame. No DFS chain, no NeRF, no per-atom error
  accumulation: each atom's position is one independent transform.
- Categorical atom features (element, formal charge, hybridization, aromatic
  flag, smallest ring size, total H count) read off RDKit.
- K=4 nearest-heavy-neighbour spherical offsets and elements as encoder hint
  features (Mol-StrucTok style "understanding" channels).

Reconstruction is a one-line spherical → Cartesian → frame-rotation. Bond
inference happens post-hoc from the reconstructed Cartesian coords.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from src.tokenizers.descriptor_schema import (
    K_NEIGHBORS,
    LIGAND_CHARGE_TO_IDX,
    LIGAND_CHARGE_VOCAB,
    LIGAND_DESCRIPTOR_DIM,
    LIGAND_ELEMENT_TO_IDX,
    LIGAND_HYBRID_OTHER_IDX,
    LIGAND_LAYOUT,
    LIGAND_NUMH_VOCAB,
    LIGAND_OTHER_IDX,
    LIGAND_RING_NONE_IDX,
    fields_by_name,
)
from src.tokenizers.geometry import (
    cartesian_to_spherical,
    cartesian_to_spherical_np,
    spherical_to_cartesian,
    spherical_to_cartesian_np,
)
from src.tokenizers.vqvae import TransformerVQVAE as LigandVQVAE  # noqa: TC001

if TYPE_CHECKING:
    from rdkit.Chem import Mol


# ---------------------------------------------------------------------------
# SDF parsing (unchanged from previous implementation)
# ---------------------------------------------------------------------------


def parse_sdf(path: str | Path) -> list[dict]:
    """Parse an SDF (or .sdf.gz) file and return one dict per molecule.

    Each molecule is a dict with keys:
    - atoms: list of (element, x, y, z)
    - bonds: list of (atom1_idx, atom2_idx, bond_type) (0-indexed)
    """
    import gzip  # noqa: PLC0415

    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt") as f:
            text = f.read()
    else:
        text = p.read_text()
    return parse_sdf_text(text)


def parse_sdf_text(text: str) -> list[dict]:  # noqa: C901, PLR0912, PLR0915
    """Parse already-decompressed SDF text into one dict per molecule.

    Same return shape as :func:`parse_sdf`; used when reading molecules from
    packed tar shards (bytes -> gunzip -> text) without extracting files.
    """
    molecules = []
    lines = text.splitlines()

    i = 0
    while i < len(lines):
        if i + 3 >= len(lines):
            break

        # Header: 3 lines (name, program/timestamp, comment)
        i += 3

        # Counts line
        counts_line = lines[i].strip()
        i += 1
        parts = counts_line.split()
        if len(parts) < 2:  # noqa: PLR2004
            while i < len(lines) and lines[i].strip() != "$$$$":
                i += 1
            i += 1
            continue

        try:
            num_atoms = int(parts[0])
            num_bonds = int(parts[1])
        except ValueError:
            while i < len(lines) and lines[i].strip() != "$$$$":
                i += 1
            i += 1
            continue

        atoms = []
        for _ in range(num_atoms):
            if i >= len(lines):
                break
            atom_line = lines[i]
            i += 1
            atom_parts = atom_line.split()
            if len(atom_parts) < 4:  # noqa: PLR2004
                continue
            x, y, z = float(atom_parts[0]), float(atom_parts[1]), float(atom_parts[2])
            element = atom_parts[3]
            atoms.append((element, x, y, z))

        bonds = []
        for _ in range(num_bonds):
            if i >= len(lines):
                break
            bond_line = lines[i]
            i += 1
            bond_parts = bond_line.split()
            if len(bond_parts) < 3:  # noqa: PLR2004
                continue
            a1 = int(bond_parts[0]) - 1
            a2 = int(bond_parts[1]) - 1
            bt = int(bond_parts[2])
            bonds.append((a1, a2, bt))

        if atoms:
            molecules.append({"atoms": atoms, "bonds": bonds})

        while i < len(lines) and lines[i].strip() != "$$$$":
            i += 1
        i += 1

    return molecules


# ---------------------------------------------------------------------------
# RDKit atom-feature extraction
# ---------------------------------------------------------------------------


def _build_rdkit_mol(
    atoms: list[tuple[str, float, float, float]],
    bonds: list[tuple[int, int, int]],
) -> Mol | None:
    """Build a sanitised RDKit Mol from parsed SDF data, or ``None`` on failure."""
    from rdkit import Chem  # noqa: PLC0415

    mol = Chem.RWMol()
    for elem, *_ in atoms:
        try:
            mol.AddAtom(Chem.Atom(elem))
        except Exception:  # noqa: BLE001
            mol.AddAtom(Chem.Atom("C"))

    bond_type_map = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    n = len(atoms)
    for a1, a2, bt in bonds:
        if 0 <= a1 < n and 0 <= a2 < n and a1 != a2:
            try:
                mol.AddBond(a1, a2, bond_type_map.get(bt, Chem.BondType.SINGLE))
            except Exception:  # noqa: BLE001, S112
                continue

    final_mol = mol.GetMol()
    try:
        Chem.SanitizeMol(final_mol)
    except Exception:  # noqa: BLE001
        # Best-effort: keep the unsanitised mol so we can still read element
        # symbols and approximate features. ``FastFindRings`` is required
        # so ``GetRingInfo`` works on the unsanitised mol; without it
        # ``IsAtomInRingOfSize`` raises "RingInfo not initialized".
        try:
            final_mol.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(final_mol)
        except Exception:  # noqa: BLE001
            return None
    return final_mol


def _atom_features_from_mol(mol: Mol, num_atoms: int) -> dict[str, np.ndarray]:
    """Extract per-atom integer features for ``num_atoms`` heavy atoms.

    Returns arrays of shape ``(num_atoms,)`` for ``element``, ``charge``,
    ``hybrid``, ``aromatic``, ``ring``, ``numH``. Missing/unsanitised atoms
    fall back to ``OTHER`` / 0 values.
    """
    from rdkit import Chem  # noqa: PLC0415

    elements = np.full(num_atoms, LIGAND_OTHER_IDX, dtype=np.int64)
    charges = np.full(num_atoms, LIGAND_CHARGE_TO_IDX[0], dtype=np.int64)
    hybrid = np.full(num_atoms, LIGAND_HYBRID_OTHER_IDX, dtype=np.int64)
    aromatic = np.zeros(num_atoms, dtype=np.int64)
    ring = np.full(num_atoms, LIGAND_RING_NONE_IDX, dtype=np.int64)
    num_h = np.zeros(num_atoms, dtype=np.int64)

    if mol is None:
        return {
            "element": elements,
            "charge": charges,
            "hybrid": hybrid,
            "aromatic": aromatic,
            "ring": ring,
            "numH": num_h,
        }

    ring_info = mol.GetRingInfo()
    hybrid_map = {
        Chem.HybridizationType.SP: 0,
        Chem.HybridizationType.SP2: 1,
        Chem.HybridizationType.SP3: 2,
    }

    for i in range(min(num_atoms, mol.GetNumAtoms())):
        atom = mol.GetAtomWithIdx(i)
        elem_sym = atom.GetSymbol()
        elements[i] = LIGAND_ELEMENT_TO_IDX.get(elem_sym, LIGAND_OTHER_IDX)

        c = max(
            min(atom.GetFormalCharge(), LIGAND_CHARGE_VOCAB[-1]), LIGAND_CHARGE_VOCAB[0]
        )
        charges[i] = LIGAND_CHARGE_TO_IDX[c]

        is_arom = bool(atom.GetIsAromatic())
        aromatic[i] = 1 if is_arom else 0
        if is_arom:
            hybrid[i] = 3  # AROM bucket
        else:
            hybrid[i] = hybrid_map.get(atom.GetHybridization(), LIGAND_HYBRID_OTHER_IDX)

        # Smallest ring containing this atom; 0 if not in any ring.
        smallest = 0
        for size in (3, 4, 5, 6):
            if ring_info.IsAtomInRingOfSize(i, size):
                smallest = size
                break
        else:
            if atom.IsInRing():
                smallest = 7  # 6+
        if smallest == 0:
            ring[i] = LIGAND_RING_NONE_IDX
        elif smallest <= 5:  # noqa: PLR2004
            ring[i] = smallest - 3  # 3->0, 4->1, 5->2
        else:
            ring[i] = 3  # 6+

        nh = atom.GetTotalNumHs(includeNeighbors=False)
        num_h[i] = max(0, min(nh, LIGAND_NUMH_VOCAB[-1]))

    return {
        "element": elements,
        "charge": charges,
        "hybrid": hybrid,
        "aromatic": aromatic,
        "ring": ring,
        "numH": num_h,
    }


# ---------------------------------------------------------------------------
# K-NN encoder hint features
# ---------------------------------------------------------------------------


def _knn_offsets_and_elements(
    canonical_coords: np.ndarray,  # (N, 3) heavy-atom coords in canonical frame
    elements: np.ndarray,  # (N,) element indices
    k: int = K_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray]:
    """For each atom, return spherical offsets + element idx of K nearest atoms.

    Distances are computed in the canonical frame; offsets are spherical
    ``(Δr, θ, sin φ, cos φ)`` of the displacement vector ``neighbour - self``.
    Self is excluded. When fewer than ``k`` other atoms exist, the trailing
    slots are zero-padded (offsets) and filled with ``OTHER`` (elements).
    """
    n = canonical_coords.shape[0]
    offsets = np.zeros((n, k * 4), dtype=np.float32)
    nbr_elements = np.full((n, k), LIGAND_OTHER_IDX, dtype=np.int64)

    if n <= 1:
        return offsets, nbr_elements

    # Pairwise distances (small N, brute force is faster than building a tree).
    diff = canonical_coords[:, None, :] - canonical_coords[None, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)

    take = min(k, n - 1)
    nbr_idx = np.argpartition(dist, take - 1, axis=1)[:, :take]
    # Order each row's neighbours by ascending distance (argpartition is unsorted).
    for i in range(n):
        order = np.argsort(dist[i, nbr_idx[i]])
        nbr_idx[i] = nbr_idx[i, order]

    for i in range(n):
        for slot, j in enumerate(nbr_idx[i]):
            delta = canonical_coords[j] - canonical_coords[i]
            r, theta, sphi, cphi = cartesian_to_spherical(delta)
            offsets[i, slot * 4 : slot * 4 + 4] = (r, theta, sphi, cphi)
            nbr_elements[i, slot] = elements[j]
    return offsets, nbr_elements


def rotate_ligand_descriptor(
    descriptor: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """Re-express a ligand descriptor under an extra frame rotation ``R``.

    The orientation-dependent slots are the absolute ``coord`` spherical
    ``(r, θ, sin φ, cos φ)`` and each of the K ``knn_offsets`` spherical
    blocks; everything else (radii are folded into those blocks, plus all
    categorical / KNN-element slots) is rotation-invariant and copied through.

    This is the cheap augmentation primitive used by GEOM pretraining: compute
    the descriptor **once** (identity frame), then synthesise each random
    orientation by rotating the stored spherical slots instead of re-running
    the RDKit feature extraction K times. Equivalent to calling
    :meth:`LigandDescriptor.compute` with ``pocket_frame=(centroid, R @ R0)``
    (verified in tests).

    Args:
        descriptor: ``(N, 30)`` descriptor produced by ``compute``.
        rotation: ``(3, 3)`` rotation matrix applied as ``p' = p @ R.T`` to
            every Cartesian (canonical) position, matching the convention in
            :meth:`LigandDescriptor.compute`.

    Returns:
        A new ``(N, 30)`` float32 descriptor; the input is not modified.
    """
    if descriptor.shape[0] == 0:
        return descriptor.copy()

    out = descriptor.astype(np.float64, copy=True)
    f = fields_by_name(LIGAND_LAYOUT)

    def _rotate_block(start: int) -> None:
        sph = out[:, start : start + 4]
        cart = spherical_to_cartesian_np(sph) @ rotation.T
        out[:, start : start + 4] = cartesian_to_spherical_np(cart)

    _rotate_block(f["coord"].start)
    knn = f["knn_offsets"]
    for k in range(knn.length // 4):
        _rotate_block(knn.start + 4 * k)

    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Descriptor class
# ---------------------------------------------------------------------------


class LigandDescriptor:
    """Spherical multi-feature per-atom descriptor with one-shot reconstruction."""

    DESCRIPTOR_DIM = LIGAND_DESCRIPTOR_DIM

    def compute(
        self,
        atoms: list[tuple[str, float, float, float]],
        bonds: list[tuple[int, int, int]],
        pocket_frame: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, list[str], dict[str, Any]]:
        """Compute descriptors for all heavy atoms.

        Args:
            atoms: ``(element, x, y, z)`` per atom (heavy + hydrogens).
            bonds: ``(atom1_idx, atom2_idx, bond_type)`` 0-indexed.
            pocket_frame: ``(centroid, rotation)`` from
                :func:`_compute_canonical_frame`. Required — the descriptor
                is only meaningful in a fixed canonical frame because it
                stores absolute spherical coords from the pocket centroid.

        Returns:
            descriptors: ``(N_heavy, 30)`` float32 array.
            elements: heavy-atom element symbols in original atom order.
            metadata: ``{'centroid', 'rotation', 'heavy_to_orig'}`` for
                inverse transform. ``heavy_to_orig[i]`` is the index of the
                i-th heavy atom in the original atom list.
        """
        if pocket_frame is None:
            msg = "LigandDescriptor.compute requires a pocket_frame"
            raise ValueError(msg)

        # Drop hydrogens up-front; bond_type still references original indices
        # but RDKit Mol is built on the heavy-only atom list (mapping below).
        heavy_indices = [i for i, (e, *_) in enumerate(atoms) if e != "H"]
        if not heavy_indices:
            empty_meta: dict[str, Any] = {
                "centroid": pocket_frame[0],
                "rotation": pocket_frame[1],
                "heavy_to_orig": [],
            }
            return (
                np.zeros((0, LIGAND_DESCRIPTOR_DIM), dtype=np.float32),
                [],
                empty_meta,
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

        # Canonical-frame coords (always required).
        centroid, rotation = pocket_frame
        canonical = (coords_global - centroid) @ rotation.T  # (N, 3)

        # Spherical (r, theta, sin phi, cos phi) per atom, from centroid (= origin).
        coord = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            r, theta, sphi, cphi = cartesian_to_spherical(canonical[i])
            coord[i] = (r, theta, sphi, cphi)

        # RDKit-derived categorical features.
        mol = _build_rdkit_mol(heavy_atoms, heavy_bonds)
        feats = _atom_features_from_mol(mol, n)
        elements = feats["element"]

        # K-NN encoder hint features, computed in canonical Cartesian space.
        knn_offsets, knn_elements = _knn_offsets_and_elements(canonical, elements)

        descriptor = np.zeros((n, LIGAND_DESCRIPTOR_DIM), dtype=np.float32)
        f = fields_by_name(LIGAND_LAYOUT)
        descriptor[:, f["coord"].start : f["coord"].end] = coord
        descriptor[:, f["element"].start] = elements
        descriptor[:, f["charge"].start] = feats["charge"]
        descriptor[:, f["hybrid"].start] = feats["hybrid"]
        descriptor[:, f["aromatic"].start] = feats["aromatic"]
        descriptor[:, f["ring"].start] = feats["ring"]
        descriptor[:, f["numH"].start] = feats["numH"]
        descriptor[:, f["knn_offsets"].start : f["knn_offsets"].end] = knn_offsets
        descriptor[:, f["knn_elements"].start : f["knn_elements"].end] = knn_elements

        metadata: dict[str, Any] = {
            "centroid": centroid,
            "rotation": rotation,
            "heavy_to_orig": heavy_indices,
        }
        return descriptor, elements_sym, metadata

    # ---- inverse transform ------------------------------------------------

    @staticmethod
    def descriptor_to_coords(
        descriptors: np.ndarray,
        metadata: dict[str, Any],
        pocket_frame: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Reconstruct global Cartesian coordinates from descriptors.

        ``pocket_frame`` overrides ``metadata`` if both are given (handy for
        cross-frame visualisation). When neither is supplied, raises.

        Returns:
            coords: ``(N_heavy, 3)`` float64 in the **global** frame.
        """
        frame = pocket_frame or (metadata["centroid"], metadata["rotation"])
        centroid, rotation = frame

        n = descriptors.shape[0]
        if n == 0:
            return np.zeros((0, 3), dtype=np.float64)

        f = fields_by_name(LIGAND_LAYOUT)
        coord = descriptors[:, f["coord"].start : f["coord"].end].astype(np.float64)

        canonical = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            r, theta, sphi, cphi = coord[i]
            canonical[i] = spherical_to_cartesian(
                float(r), float(theta), float(sphi), float(cphi)
            )
        return canonical @ rotation + centroid


def solve_ligand_coords(  # noqa: PLR0913
    coord_sph: np.ndarray,  # (N, 4)  decoded absolute spherical (canonical frame)
    knn_offsets_sph: np.ndarray,  # (N, 4*K)  decoded K-NN relative spherical offsets
    pocket_frame: tuple[np.ndarray, np.ndarray],
    *,
    k: int = K_NEIGHBORS,
    n_steps: int = 50,
    lr: float = 0.05,
    d_floor: float = 1.1,
    w_abs: float = 1.0,
    w_rel: float = 5.0,
    w_clash: float = 1.0,
) -> np.ndarray:
    """Reconstruct global Cartesian coords by solving a small geometry problem.

    Instead of trusting the per-atom absolute coord head alone (pairwise-blind →
    frequent clashes), we minimise a weighted objective over atom positions ``X``:

    - ``w_abs``   * |X - X_abs|^2                  weak global anchor (placement)
    - ``w_rel``   * |(X[nbr] - X) - offset_pred|^2 strong local geometry
    - ``w_clash`` * relu(d_floor - |X_i - X_j|)^2  hard no-overlap

    Neighbour correspondence ``nbr`` is the K nearest atoms in the decoded
    absolute coords (matching how the descriptor's K-NN were built). The solve is
    a few Adam steps; weights/steps are decode-time knobs (no retraining needed).
    """
    centroid, rotation = pocket_frame
    n = coord_sph.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    abs_cart = spherical_to_cartesian_np(coord_sph)  # (N, 3) canonical
    if n <= 2:  # noqa: PLR2004
        return abs_cart @ rotation + centroid

    off = knn_offsets_sph.reshape(n, -1, 4)[:, :k, :]
    off_cart = np.stack(
        [spherical_to_cartesian_np(off[:, j, :]) for j in range(off.shape[1])],
        axis=1,
    )  # (N, k, 3)

    dist = np.linalg.norm(abs_cart[:, None, :] - abs_cart[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    take = min(k, n - 1)
    nbr = np.argsort(dist, axis=1)[:, :take]  # (N, take)

    a = torch.tensor(abs_cart, dtype=torch.float32)
    off_t = torch.tensor(off_cart[:, :take, :], dtype=torch.float32)  # (N, take, 3)
    nbr_t = torch.tensor(nbr, dtype=torch.long)
    offdiag = ~torch.eye(n, dtype=torch.bool)
    # Re-enable grad: this optimisation runs inside ``_decode_ligand``'s
    # ``@torch.no_grad()`` context, but the solve itself needs autograd.
    with torch.enable_grad():
        x = a.clone().requires_grad_(True)  # noqa: FBT003
        opt = torch.optim.Adam([x], lr=lr)
        for _ in range(n_steps):
            opt.zero_grad()
            l_abs = (x - a).pow(2).sum(-1).mean()
            rel = x[nbr_t] - x.unsqueeze(1)  # (N, take, 3)
            l_rel = (rel - off_t).pow(2).sum(-1).mean()
            pair = torch.cdist(x, x)
            l_clash = torch.relu(d_floor - pair)[offdiag].pow(2).mean()
            loss = w_abs * l_abs + w_rel * l_rel + w_clash * l_clash
            loss.backward()
            opt.step()
    solved = x.detach().numpy().astype(np.float64)
    return solved @ rotation + centroid


class LigandTokenizer:
    """High-level ligand tokenizer that emits raw VQ-VAE codebook indices.

    Unlike the previous element-prefixed format (``"C_20"``), the codebook
    index alone now carries the element + atom features (recoverable from
    the VQ-VAE decoder), so each atom maps to a single integer token.
    """

    def __init__(self, vqvae: LigandVQVAE) -> None:
        self.vqvae = vqvae
        self.descriptor = LigandDescriptor()

    def tokenize(
        self,
        atoms: list[tuple[str, float, float, float]],
        bonds: list[tuple[int, int, int]],
        pocket_frame: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> list[int]:
        descriptors, _elements, _metadata = self.descriptor.compute(
            atoms,
            bonds,
            pocket_frame=pocket_frame,
        )
        if len(descriptors) == 0:
            return []

        desc_tensor = torch.from_numpy(descriptors).to(
            next(self.vqvae.parameters()).device,
        )
        with torch.no_grad():
            indices = self.vqvae.encode(desc_tensor)
        return indices.cpu().tolist()

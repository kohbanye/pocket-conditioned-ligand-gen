"""Ligand 3D structure tokenization using invertible Z-matrix descriptors.

Provides:
- Canonical DFS ordering via RDKit
- Z-matrix internal coordinates (4D per atom)
- Inverse reconstruction via NeRF algorithm
- VQ-VAE for structure tokenization
- SDF file parsing
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor, nn

from src.tokenizers.codebook import EMACodebook

_EPS = 1e-8

if TYPE_CHECKING:
    from src.config import LigandVQVAEConfig


def parse_sdf(path: str | Path) -> list[dict]:  # noqa: C901, PLR0912, PLR0915
    """Parse an SDF file and return a list of molecules.

    Supports both plain ``.sdf`` and gzipped ``.sdf.gz`` files.

    Each molecule is a dict with keys:
    - atoms: list of (element, x, y, z)
    - bonds: list of (atom1_idx, atom2_idx, bond_type) (0-indexed)
    """
    import gzip  # noqa: PLC0415

    molecules = []
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt") as f:
            lines = f.read().splitlines()
    else:
        lines = p.read_text().splitlines()

    i = 0
    while i < len(lines):
        # Skip to counts line (line 4 of each mol block, index 3)
        if i + 3 >= len(lines):
            break

        # Header: 3 lines (name, program/timestamp, comment)
        i += 3

        # Counts line
        counts_line = lines[i].strip()
        i += 1
        parts = counts_line.split()
        if len(parts) < 2:  # noqa: PLR2004
            # Skip to next $$$$ delimiter
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

        # Atom block
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

        # Bond block
        bonds = []
        for _ in range(num_bonds):
            if i >= len(lines):
                break
            bond_line = lines[i]
            i += 1
            bond_parts = bond_line.split()
            if len(bond_parts) < 3:  # noqa: PLR2004
                continue
            a1 = int(bond_parts[0]) - 1  # Convert to 0-indexed
            a2 = int(bond_parts[1]) - 1
            bt = int(bond_parts[2])
            bonds.append((a1, a2, bt))

        if atoms:
            molecules.append({"atoms": atoms, "bonds": bonds})

        # Skip to $$$$
        while i < len(lines) and lines[i].strip() != "$$$$":
            i += 1
        i += 1

    return molecules


def _build_adjacency(
    num_atoms: int,
    bonds: list[tuple[int, int, int]],
) -> list[list[int]]:
    """Build adjacency list from bond information."""
    adj: list[list[int]] = [[] for _ in range(num_atoms)]
    for a1, a2, _bt in bonds:
        if 0 <= a1 < num_atoms and 0 <= a2 < num_atoms:
            adj[a1].append(a2)
            adj[a2].append(a1)
    return adj


# ---------------------------------------------------------------------------
# Canonical DFS ordering
# ---------------------------------------------------------------------------


def _canonical_atom_ranks(
    atoms: list[tuple[str, float, float, float]],
    bonds: list[tuple[int, int, int]],
) -> list[int]:
    """Compute canonical atom ranks using RDKit.

    Falls back to simple atom-index ordering if RDKit Mol construction fails.
    """
    from rdkit import Chem  # noqa: PLC0415

    num_atoms = len(atoms)
    mol = Chem.RWMol()
    for elem, *_ in atoms:
        mol.AddAtom(Chem.Atom(elem))

    bond_type_map = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }
    for a1, a2, bt in bonds:
        if 0 <= a1 < num_atoms and 0 <= a2 < num_atoms:
            mol.AddBond(a1, a2, bond_type_map.get(bt, Chem.BondType.SINGLE))

    try:
        final_mol = mol.GetMol()
        final_mol.UpdatePropertyCache(strict=False)
        return list(Chem.CanonicalRankAtoms(final_mol))
    except Exception:  # noqa: BLE001
        return list(range(num_atoms))


def _canonical_dfs_order(
    num_atoms: int,
    adj: list[list[int]],
    ranks: list[int],
) -> tuple[list[int], list[int], list[tuple[int, int, int]]]:
    """Canonical DFS traversal based on atom ranks.

    Returns:
        order: atom indices in DFS visit order.
        tree_parent: ``tree_parent[atom_idx]`` = parent atom idx (-1 for roots).
        ring_closures: back-edge bonds ``(a1, a2, bond_type)`` not in the
            spanning tree.  (``bond_type`` is always stored as 0 here because
            the original bond type is not available inside this helper;
            callers that need it should cross-reference with the full bond
            list.)
    """
    if num_atoms == 0:
        return [], [], []

    visited = [False] * num_atoms
    order: list[int] = []
    tree_parent = [-1] * num_atoms

    def _dfs_component(start: int) -> None:
        stack = [(start, -1)]
        while stack:
            node, parent = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            order.append(node)
            tree_parent[node] = parent
            stack.extend(
                (nbr, node)
                for nbr in sorted(adj[node], key=lambda n: ranks[n], reverse=True)
                if not visited[nbr]
            )

    # Start from the atom with the smallest canonical rank
    start = min(range(num_atoms), key=lambda i: ranks[i])
    _dfs_component(start)

    # Handle disconnected components
    for atom_idx in sorted(range(num_atoms), key=lambda i: ranks[i]):
        if not visited[atom_idx]:
            _dfs_component(atom_idx)

    return order, tree_parent, []  # ring_closures filled by caller


# ---------------------------------------------------------------------------
# Z-matrix reference selection
# ---------------------------------------------------------------------------


def _compute_zmat_refs(
    order: list[int],
    tree_parent: list[int],
    adj: list[list[int]],
) -> list[tuple[int, int, int]]:
    """Determine (parent, angle_ref, dihedral_ref) for each DFS position.

    All values are *original atom indices*.  ``-1`` means not available.
    """
    placed: set[int] = set()
    refs: list[tuple[int, int, int]] = []

    for pos, atom in enumerate(order):
        if pos == 0:
            refs.append((-1, -1, -1))
            placed.add(atom)
            continue

        parent = tree_parent[atom]

        if pos == 1:
            refs.append((parent, -1, -1))
            placed.add(atom)
            continue

        # --- angle reference: grandparent preferred -------------------------
        grandparent = tree_parent[parent]
        if grandparent != -1:
            angle_ref = grandparent
        else:
            angle_ref = _find_placed_neighbor(adj[parent], placed, exclude={atom})

        if angle_ref == -1:
            refs.append((parent, -1, -1))
            placed.add(atom)
            continue

        # --- dihedral reference: great-grandparent or other placed atom -----
        dihedral_ref = _find_dihedral_ref(
            tree_parent,
            adj,
            angle_ref,
            parent,
            atom,
            placed,
        )

        refs.append((parent, angle_ref, dihedral_ref))
        placed.add(atom)

    return refs


def _find_placed_neighbor(
    neighbors: list[int],
    placed: set[int],
    *,
    exclude: set[int],
) -> int:
    """Return the first placed neighbor not in *exclude*, or -1."""
    for n in neighbors:
        if n not in exclude and n in placed:
            return n
    return -1


def _find_dihedral_ref(  # noqa: PLR0913
    tree_parent: list[int],
    adj: list[list[int]],
    angle_ref: int,
    parent: int,
    atom: int,
    placed: set[int],
) -> int:
    """Find a suitable dihedral reference atom.

    The returned atom must differ from *parent*, *angle_ref*, and *atom*
    to avoid a degenerate four-point dihedral.
    """
    excluded = {parent, angle_ref, atom}
    ggp = tree_parent[angle_ref]
    if ggp != -1 and ggp not in excluded:
        return ggp
    ref = _find_placed_neighbor(adj[angle_ref], placed, exclude=excluded)
    if ref != -1:
        return ref
    return _find_placed_neighbor(adj[parent], placed, exclude=excluded)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _bond_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the angle (radians) at *b* in the triangle a-b-c."""
    v1 = a - b
    v2 = c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < _EPS:
        return 0.0
    cos_angle = np.dot(v1, v2) / denom
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _dihedral_angle(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> float:
    """Dihedral angle (radians) defined by four points p0-p1-p2-p3."""
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

    return float(np.arctan2(np.dot(m, n2), np.dot(n1, n2)))


def _place_atom(  # noqa: PLR0913
    ref_a: np.ndarray,
    ref_b: np.ndarray,
    ref_c: np.ndarray,
    bond_length: float,
    bond_angle: float,
    torsion: float,
) -> np.ndarray:
    """Place atom D via NeRF so that ``dihedral(A, B, C, D) == torsion``.

    *ref_a*, *ref_b*, *ref_c* correspond to dihedral_ref, angle_ref, parent.
    """
    v1 = ref_b - ref_c
    v1_norm = np.linalg.norm(v1)
    if v1_norm < _EPS:
        return ref_c + np.array([bond_length, 0.0, 0.0])
    v1_hat = v1 / v1_norm

    v2 = ref_a - ref_b
    n = np.cross(v1_hat, v2)
    n_norm = np.linalg.norm(n)

    if n_norm < _EPS:
        perp = (
            np.array([1.0, 0.0, 0.0])
            if abs(v1_hat[0]) < 0.9  # noqa: PLR2004
            else np.array([0.0, 1.0, 0.0])
        )
        n = np.cross(v1_hat, perp)
        n = n / np.linalg.norm(n)
    else:
        n = n / n_norm

    m = np.cross(n, v1_hat)

    return ref_c + bond_length * (
        np.cos(bond_angle) * v1_hat
        + np.sin(bond_angle) * np.cos(torsion) * m
        + np.sin(bond_angle) * np.sin(torsion) * n
    )


def _virtual_dihedral_ref(
    angle_ref_pos: np.ndarray,
    parent_pos: np.ndarray,
) -> np.ndarray:
    """Construct a deterministic virtual dihedral reference point.

    Used when no real dihedral reference is available (DFS positions 0-2 and
    disconnected-component starts).  The virtual point is placed so that
    ``torsion = 0`` produces a deterministic, reproducible placement.
    """
    v = angle_ref_pos - parent_pos
    v_hat = v / (np.linalg.norm(v) + _EPS)
    up = (
        np.array([0.0, 1.0, 0.0])
        if abs(v_hat[1]) < 0.9  # noqa: PLR2004
        else np.array([0.0, 0.0, 1.0])
    )
    perp = up - np.dot(up, v_hat) * v_hat
    perp = perp / (np.linalg.norm(perp) + _EPS)
    return angle_ref_pos + perp


def _canonical_virtual_ref(
    angle_ref_pos: np.ndarray,
    parent_pos: np.ndarray,
) -> np.ndarray:
    """Virtual dihedral ref aligned with the canonical frame z-axis.

    Used in pocket-anchored mode so that the torsion encodes
    orientation relative to the pocket's canonical frame.
    Falls back to x-axis if the bond is nearly parallel to z.
    """
    bond = parent_pos - angle_ref_pos
    bond_norm = np.linalg.norm(bond)
    if bond_norm < _EPS:
        return angle_ref_pos + np.array([0.0, 0.0, 1.0])
    bond_hat = bond / bond_norm
    axis = (
        np.array([1.0, 0.0, 0.0])
        if abs(bond_hat[2]) > 0.9  # noqa: PLR2004
        else np.array([0.0, 0.0, 1.0])
    )
    return angle_ref_pos + axis


# ---------------------------------------------------------------------------
# Spherical coordinate helpers (for pocket-anchored mode)
# ---------------------------------------------------------------------------


def _cartesian_to_spherical(
    pos: np.ndarray,
) -> tuple[float, float, float, float]:
    """Convert 3D Cartesian to ``(r, θ_polar, sin φ, cos φ)``."""
    r = float(np.linalg.norm(pos))
    if r < _EPS:
        return 0.0, 0.0, 0.0, 1.0
    theta = float(np.arccos(np.clip(pos[2] / r, -1.0, 1.0)))
    phi = float(np.arctan2(pos[1], pos[0]))
    return r, theta, float(np.sin(phi)), float(np.cos(phi))


def _spherical_to_cartesian(
    r: float,
    theta: float,
    sin_phi: float,
    cos_phi: float,
) -> np.ndarray:
    """Convert ``(r, θ_polar, sin φ, cos φ)`` to 3D Cartesian."""
    phi = np.arctan2(sin_phi, cos_phi)
    return np.array(
        [
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ]
    )


# ---------------------------------------------------------------------------
# Ring-closure identification
# ---------------------------------------------------------------------------


def _find_ring_closures(
    num_atoms: int,
    tree_parent: list[int],
    bonds: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """Return bonds that are *not* part of the DFS spanning tree."""
    tree_edges: set[tuple[int, int]] = set()
    for atom_idx in range(num_atoms):
        p = tree_parent[atom_idx]
        if p != -1:
            tree_edges.add((min(atom_idx, p), max(atom_idx, p)))

    closures = []
    for a1, a2, bt in bonds:
        edge = (min(a1, a2), max(a1, a2))
        if edge not in tree_edges:
            closures.append((a1, a2, bt))
    return closures


# ---------------------------------------------------------------------------
# Descriptor class
# ---------------------------------------------------------------------------


class LigandDescriptor:
    """Z-matrix per-atom descriptor with analytical inverse.

    Each atom receives a **4-D** descriptor.  The meaning of the four
    components depends on whether the descriptor is *pocket-anchored*:

    **Without pocket frame** (standalone mode):

    *   ``(bond_length, bond_angle, sin_dihedral, cos_dihedral)``
    *   Root / early atoms get padding values.

    **With pocket frame** (anchored mode):

    *   *pos 0* (root): ``(r, θ_polar, sin φ, cos φ)`` — spherical
        coordinates of the root atom relative to the pocket centroid.
    *   *pos 1*: ``(d, θ_dir, sin φ_dir, cos φ_dir)`` — bond length +
        direction from root in the canonical frame.
    *   *pos 2* (no dihedral ref): ``(d, θ_bond, sin τ, cos τ)`` —
        torsion measured against the canonical frame z-axis.
    *   *pos 3+*: standard Z-matrix (unchanged).

    This anchors the ligand's 6-DOF pose in the pocket's canonical
    coordinate system while keeping the rest of the molecule in
    frame-invariant internal coordinates.
    """

    DESCRIPTOR_DIM = 4

    def compute(  # noqa: PLR0912, PLR0915
        self,
        atoms: list[tuple[str, float, float, float]],
        bonds: list[tuple[int, int, int]],
        pocket_frame: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, list[str], dict[str, Any]]:
        """Compute Z-matrix descriptors for every atom.

        Args:
            atoms: List of (element, x, y, z) tuples.
            bonds: List of (atom1_idx, atom2_idx, bond_type) tuples.
            pocket_frame: Optional ``(centroid, rotation)`` from
                :class:`CanonicalPocketDescriptor`.  When provided the
                first atoms encode the ligand's pose in the pocket's
                canonical frame.

        Returns:
            descriptors: ``(N, 4)`` float32 array in canonical DFS order.
            elements: element symbols in canonical DFS order.
            metadata: dict needed for coordinate reconstruction.
        """
        num_atoms = len(atoms)
        empty_meta: dict[str, Any] = {
            "order": [],
            "refs": [],
            "ring_closures": [],
            "anchored": False,
        }
        if num_atoms == 0:
            return np.zeros((0, 4), dtype=np.float32), [], empty_meta

        coords = np.array([(a[1], a[2], a[3]) for a in atoms], dtype=np.float64)
        all_elements = [a[0] for a in atoms]
        adj = _build_adjacency(num_atoms, bonds)

        ranks = _canonical_atom_ranks(atoms, bonds)
        order, tree_parent, _ = _canonical_dfs_order(num_atoms, adj, ranks)
        refs = _compute_zmat_refs(order, tree_parent, adj)
        ring_closures = _find_ring_closures(num_atoms, tree_parent, bonds)

        anchored = pocket_frame is not None
        if anchored:
            centroid, rotation = pocket_frame
            wc = (coords - centroid) @ rotation.T  # canonical frame
        else:
            wc = coords

        descriptors = np.zeros((num_atoms, 4), dtype=np.float64)

        for pos, atom in enumerate(order):
            parent_idx, angle_ref_idx, dihedral_ref_idx = refs[pos]

            # --- root atom (or start of disconnected component) -------------
            if parent_idx == -1:
                if anchored:
                    descriptors[pos] = _cartesian_to_spherical(wc[atom])
                else:
                    descriptors[pos, 3] = 1.0
                continue

            d = float(np.linalg.norm(wc[atom] - wc[parent_idx]))
            descriptors[pos, 0] = d

            # --- second atom in component (no angle ref) --------------------
            if angle_ref_idx == -1:
                if anchored and d >= _EPS:
                    delta_hat = (wc[atom] - wc[parent_idx]) / d
                    _, th, sp, cp = _cartesian_to_spherical(delta_hat)
                    descriptors[pos, 1] = th
                    descriptors[pos, 2] = sp
                    descriptors[pos, 3] = cp
                else:
                    descriptors[pos, 3] = 1.0
                continue

            theta = _bond_angle(wc[angle_ref_idx], wc[parent_idx], wc[atom])
            descriptors[pos, 1] = theta

            # --- no dihedral ref: virtual reference -------------------------
            if dihedral_ref_idx == -1:
                if anchored:
                    virtual = _canonical_virtual_ref(
                        wc[angle_ref_idx],
                        wc[parent_idx],
                    )
                    tau = _dihedral_angle(
                        virtual,
                        wc[angle_ref_idx],
                        wc[parent_idx],
                        wc[atom],
                    )
                    descriptors[pos, 2] = np.sin(tau)
                    descriptors[pos, 3] = np.cos(tau)
                else:
                    descriptors[pos, 3] = 1.0
                continue

            # --- standard Z-matrix dihedral ---------------------------------
            tau = _dihedral_angle(
                wc[dihedral_ref_idx],
                wc[angle_ref_idx],
                wc[parent_idx],
                wc[atom],
            )
            descriptors[pos, 2] = np.sin(tau)
            descriptors[pos, 3] = np.cos(tau)

        elements = [all_elements[i] for i in order]

        metadata: dict[str, Any] = {
            "order": order,
            "refs": refs,
            "ring_closures": ring_closures,
            "anchored": anchored,
        }
        return descriptors.astype(np.float32), elements, metadata

    # ---- inverse transform ------------------------------------------------

    @staticmethod
    def descriptor_to_coords(
        descriptors: np.ndarray,
        metadata: dict[str, Any],
        pocket_frame: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Reconstruct Cartesian coordinates from Z-matrix descriptors.

        When ``metadata['anchored']`` is True the reconstruction places
        atoms in the pocket's canonical frame.  If *pocket_frame* is also
        supplied the result is transformed back to the global frame.

        Returns:
            coords: ``(N, 3)`` array in **original** atom order.
        """
        order: list[int] = metadata["order"]
        refs: list[tuple[int, int, int]] = metadata["refs"]
        anchored: bool = metadata.get("anchored", False)
        n_atoms = len(order)

        if n_atoms == 0:
            return np.zeros((0, 3), dtype=np.float64)

        coords = np.zeros((n_atoms, 3), dtype=np.float64)

        for pos in range(n_atoms):
            atom_idx = order[pos]
            parent_idx, angle_ref_idx, dihedral_ref_idx = refs[pos]

            d = float(descriptors[pos, 0])
            theta = float(descriptors[pos, 1])
            sin_tau = float(descriptors[pos, 2])
            cos_tau = float(descriptors[pos, 3])

            # --- root atom --------------------------------------------------
            if parent_idx == -1:
                if anchored:
                    coords[atom_idx] = _spherical_to_cartesian(
                        d,
                        theta,
                        sin_tau,
                        cos_tau,
                    )
                else:
                    coords[atom_idx] = [0.0, 0.0, 0.0]

            # --- second atom (no angle ref) ---------------------------------
            elif angle_ref_idx == -1:
                if anchored:
                    direction = _spherical_to_cartesian(
                        1.0,
                        theta,
                        sin_tau,
                        cos_tau,
                    )
                    coords[atom_idx] = coords[parent_idx] + d * direction
                else:
                    coords[atom_idx] = coords[parent_idx] + np.array(
                        [d, 0.0, 0.0],
                    )

            # --- no dihedral ref: virtual reference -------------------------
            elif dihedral_ref_idx == -1:
                virtual = (
                    _canonical_virtual_ref(
                        coords[angle_ref_idx],
                        coords[parent_idx],
                    )
                    if anchored
                    else _virtual_dihedral_ref(
                        coords[angle_ref_idx],
                        coords[parent_idx],
                    )
                )
                tau = float(np.arctan2(sin_tau, cos_tau))
                coords[atom_idx] = _place_atom(
                    virtual,
                    coords[angle_ref_idx],
                    coords[parent_idx],
                    d,
                    theta,
                    tau,
                )

            # --- standard Z-matrix ------------------------------------------
            else:
                tau = float(np.arctan2(sin_tau, cos_tau))
                coords[atom_idx] = _place_atom(
                    coords[dihedral_ref_idx],
                    coords[angle_ref_idx],
                    coords[parent_idx],
                    d,
                    theta,
                    tau,
                )

        # Transform from canonical frame to global if requested
        if pocket_frame is not None:
            centroid, rotation = pocket_frame
            for i in range(n_atoms):
                coords[i] = coords[i] @ rotation + centroid

        return coords


# ---------------------------------------------------------------------------
# VQ-VAE
# ---------------------------------------------------------------------------


class LigandVQVAE(nn.Module):
    """VQ-VAE for ligand atom structure tokenization."""

    def __init__(self, config: LigandVQVAEConfig) -> None:
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


class LigandTokenizer:
    """High-level ligand tokenizer combining descriptors and VQ-VAE."""

    def __init__(self, vqvae: LigandVQVAE) -> None:
        self.vqvae = vqvae
        self.descriptor = LigandDescriptor()

    def tokenize(
        self,
        atoms: list[tuple[str, float, float, float]],
        bonds: list[tuple[int, int, int]],
        pocket_frame: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> list[str]:
        """Tokenize a molecule into element_code tokens.

        Args:
            atoms: List of (element, x, y, z) tuples.
            bonds: List of (atom1_idx, atom2_idx, bond_type) tuples.
            pocket_frame: Optional ``(centroid, rotation)`` for anchoring.

        Returns:
            List of tokens like ["C_20", "O_23", "N_6"].
        """
        descriptors, elements, _metadata = self.descriptor.compute(
            atoms,
            bonds,
            pocket_frame=pocket_frame,
        )
        if len(descriptors) == 0:
            return []

        desc_tensor = torch.from_numpy(descriptors)
        device = next(self.vqvae.parameters()).device
        desc_tensor = desc_tensor.to(device)

        with torch.no_grad():
            indices = self.vqvae.encode(desc_tensor)

        indices_list = indices.cpu().tolist()
        return [
            f"{elem}_{code}" for elem, code in zip(elements, indices_list, strict=True)
        ]

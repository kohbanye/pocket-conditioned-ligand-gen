"""Ligand 3D structure tokenization based on Mol-StrucTok.

Provides:
- SE(3)-invariant per-atom descriptors (14D)
- VQ-VAE for structure tokenization
- SDF file parsing
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

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
    num_atoms: int, bonds: list[tuple[int, int, int]],
) -> list[list[int]]:
    """Build adjacency list from bond information."""
    adj: list[list[int]] = [[] for _ in range(num_atoms)]
    for a1, a2, _bt in bonds:
        if 0 <= a1 < num_atoms and 0 <= a2 < num_atoms:
            adj[a1].append(a2)
            adj[a2].append(a1)
    return adj


class SE3InvariantDescriptor:
    """Compute SE(3)-invariant per-atom descriptors following Mol-StrucTok.

    Each atom gets a 14D descriptor:
    - Generation (4D): distance to focal atom, polar angle, |azimuthal|, sign(azimuthal)
    - Understanding (10D): 4 nearest-neighbor bond lengths + 6 pairwise angles
    """

    def __init__(self, max_neighbors: int = 4) -> None:
        self.max_neighbors = max_neighbors

    def compute(
        self,
        atoms: list[tuple[str, float, float, float]],
        bonds: list[tuple[int, int, int]],
    ) -> tuple[np.ndarray, list[str]]:
        """Compute descriptors for all atoms in a molecule.

        Args:
            atoms: List of (element, x, y, z) tuples.
            bonds: List of (atom1_idx, atom2_idx, bond_type) tuples.

        Returns:
            Tuple of (descriptors, elements) where descriptors has shape
            ``(N, 14)`` and elements is a list of element symbols.
        """
        num_atoms = len(atoms)
        if num_atoms == 0:
            return np.zeros((0, 14), dtype=np.float32), []

        coords = np.array([(a[1], a[2], a[3]) for a in atoms], dtype=np.float64)
        elements = [a[0] for a in atoms]
        adj = _build_adjacency(num_atoms, bonds)

        bfs_order = self._bfs_order(num_atoms, adj)
        bfs_rank = np.zeros(num_atoms, dtype=int)
        for rank, atom_idx in enumerate(bfs_order):
            bfs_rank[atom_idx] = rank

        descriptors = np.zeros((num_atoms, 14), dtype=np.float64)

        for rank, atom_idx in enumerate(bfs_order):
            gen_desc = self._generation_descriptor(
                atom_idx, rank, bfs_order, bfs_rank, coords, adj,
            )
            und_desc = self._understanding_descriptor(atom_idx, coords, adj)
            descriptors[atom_idx, :4] = gen_desc
            descriptors[atom_idx, 4:] = und_desc

        return descriptors.astype(np.float32), elements

    def _bfs_order(self, num_atoms: int, adj: list[list[int]]) -> list[int]:  # noqa: C901
        """Compute BFS traversal order starting from atom 0."""
        if num_atoms == 0:
            return []

        visited = [False] * num_atoms
        order: list[int] = []
        queue: deque[int] = deque()

        # Start from atom 0
        queue.append(0)
        visited[0] = True

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in sorted(adj[node]):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        # Handle disconnected components
        for atom_idx in range(num_atoms):
            if not visited[atom_idx]:
                visited[atom_idx] = True
                order.append(atom_idx)
                queue.append(atom_idx)
                while queue:
                    node = queue.popleft()
                    if node != atom_idx:
                        order.append(node)
                    for neighbor in sorted(adj[node]):
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)

        return order

    def _find_focal_atom(
        self,
        atom_idx: int,
        bfs_order: list[int],
        bfs_rank: np.ndarray,
        adj: list[list[int]],
    ) -> int | None:
        """Find the topologically closest predecessor in BFS order."""
        rank = bfs_rank[atom_idx]
        if rank == 0:
            return None

        # Among bonded neighbors, find the one with smallest BFS rank
        # that comes before this atom in BFS order
        best = None
        best_rank = rank
        for neighbor in adj[atom_idx]:
            if bfs_rank[neighbor] < best_rank:
                best = neighbor
                best_rank = bfs_rank[neighbor]

        # Fallback: any predecessor in BFS order
        if best is None:
            best = bfs_order[rank - 1]

        return best

    def _generation_descriptor(  # noqa: PLR0913
        self,
        atom_idx: int,
        rank: int,
        bfs_order: list[int],
        bfs_rank: np.ndarray,
        coords: np.ndarray,
        adj: list[list[int]],
    ) -> np.ndarray:
        """Compute 4D generation descriptor (spherical coords relative to focal).

        Returns [distance, polar_angle, |azimuthal|, sign(azimuthal)].
        For the first few atoms in BFS where a proper reference frame
        cannot be built, only the distance is set (angles remain zero).
        """
        desc = np.zeros(4, dtype=np.float64)

        if rank < 1:
            return desc

        focal = self._find_focal_atom(atom_idx, bfs_order, bfs_rank, adj)
        if focal is None:
            return desc

        focal_pos = coords[focal]
        atom_pos = coords[atom_idx]
        rel = atom_pos - focal_pos
        dist = np.linalg.norm(rel)
        desc[0] = dist

        if dist < _EPS:
            return desc

        # Build reference frame — returns None if not enough predecessors
        frame = self._build_reference_frame(focal, bfs_order, bfs_rank, coords, adj)
        if frame is None:
            return desc

        # Project into local frame
        local = frame @ rel  # (3,)

        # Spherical coordinates
        r = np.linalg.norm(local)
        if r < _EPS:
            return desc

        theta = np.arccos(np.clip(local[2] / r, -1.0, 1.0))  # polar
        phi = np.arctan2(local[1], local[0])  # azimuthal

        desc[1] = theta
        desc[2] = abs(phi)
        desc[3] = np.sign(phi)

        return desc

    def _build_reference_frame(
        self,
        atom_idx: int,
        bfs_order: list[int],
        bfs_rank: np.ndarray,
        coords: np.ndarray,
        adj: list[list[int]],
    ) -> np.ndarray | None:
        """Build orthonormal reference frame for an atom via Gram-Schmidt.

        Uses only predecessor atoms in BFS order (no fixed fallback vectors).
        Returns ``None`` if fewer than 2 non-collinear predecessors exist.
        """
        rank = bfs_rank[atom_idx]
        pos = coords[atom_idx]

        # Collect predecessor atoms (bonded neighbors with lower rank first,
        # then any atoms with lower rank)
        predecessors = sorted(
            [n for n in adj[atom_idx] if bfs_rank[n] < rank],
            key=lambda n: bfs_rank[n],
        )

        # Supplement with non-bonded predecessors if needed
        if len(predecessors) < 2:  # noqa: PLR2004
            for r in range(rank - 1, -1, -1):
                prev = bfs_order[r]
                if prev not in predecessors:
                    predecessors.append(prev)
                if len(predecessors) >= 2:  # noqa: PLR2004
                    break

        if len(predecessors) < 2:  # noqa: PLR2004
            return None

        v1 = coords[predecessors[0]] - pos
        v1_norm = np.linalg.norm(v1)
        if v1_norm < _EPS:
            return None
        e1 = v1 / v1_norm

        v2 = coords[predecessors[1]] - pos
        # Gram-Schmidt: remove component along e1
        v2_proj = v2 - np.dot(v2, e1) * e1
        v2_norm = np.linalg.norm(v2_proj)
        if v2_norm < _EPS:
            # Collinear predecessors — try further predecessors
            for p in predecessors[2:]:
                v2 = coords[p] - pos
                v2_proj = v2 - np.dot(v2, e1) * e1
                v2_norm = np.linalg.norm(v2_proj)
                if v2_norm >= _EPS:
                    break
            else:
                return None

        e2 = v2_proj / v2_norm
        e3 = np.cross(e1, e2)

        return np.stack([e1, e2, e3])

    def _understanding_descriptor(
        self,
        atom_idx: int,
        coords: np.ndarray,
        adj: list[list[int]],
    ) -> np.ndarray:
        """Compute 10D understanding descriptor.

        Uses bonded neighbors (from molecular graph) for deterministic,
        SE(3)-invariant features.  4 sorted bond lengths + 6 sorted
        pairwise angles.
        """
        desc = np.zeros(10, dtype=np.float64)
        neighbors = adj[atom_idx]

        if not neighbors:
            return desc

        # Compute distances to bonded neighbors and sort
        neighbor_dists = [
            (np.linalg.norm(coords[n] - coords[atom_idx]), n) for n in neighbors
        ]
        neighbor_dists.sort()  # sort by distance

        # Take up to max_neighbors
        selected = neighbor_dists[: self.max_neighbors]

        # Sorted bond lengths
        for i, (d, _n) in enumerate(selected):
            desc[i] = d

        # Pairwise angles between bond vectors, sorted
        angles = []
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                vi = coords[selected[i][1]] - coords[atom_idx]
                vj = coords[selected[j][1]] - coords[atom_idx]
                vi_norm = np.linalg.norm(vi)
                vj_norm = np.linalg.norm(vj)
                if vi_norm > _EPS and vj_norm > _EPS:
                    cos_angle = np.dot(vi, vj) / (vi_norm * vj_norm)
                    angles.append(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                else:
                    angles.append(0.0)

        angles.sort()
        for i, angle in enumerate(angles[:6]):
            desc[4 + i] = angle

        return desc


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

    def __init__(self, vqvae: LigandVQVAE, max_neighbors: int = 4) -> None:
        self.vqvae = vqvae
        self.descriptor = SE3InvariantDescriptor(max_neighbors=max_neighbors)

    def tokenize(
        self,
        atoms: list[tuple[str, float, float, float]],
        bonds: list[tuple[int, int, int]],
    ) -> list[str]:
        """Tokenize a molecule into element_code tokens.

        Args:
            atoms: List of (element, x, y, z) tuples.
            bonds: List of (atom1_idx, atom2_idx, bond_type) tuples.

        Returns:
            List of tokens like ["C_20", "O_23", "N_6"].
        """
        descriptors, elements = self.descriptor.compute(atoms, bonds)
        if len(descriptors) == 0:
            return []

        desc_tensor = torch.from_numpy(descriptors)
        device = next(self.vqvae.parameters()).device
        desc_tensor = desc_tensor.to(device)

        with torch.no_grad():
            indices = self.vqvae.encode(desc_tensor)

        indices_list = indices.cpu().tolist()
        return [
            f"{elem}_{code}"
            for elem, code in zip(elements, indices_list, strict=True)
        ]

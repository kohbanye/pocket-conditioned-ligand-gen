"""Write protein-ligand complexes to PDB, and perceive ligand bonds by distance.

Used wherever generated or reconstructed coordinates have to leave the model as
a file a viewer or a docking program can read: ligand generation, pose export,
and the reconstruction benchmark.

Bond perception here is deliberately geometric (covalent radii + tolerance)
rather than chemical: the models emit coordinates and element labels, not a
bond graph, so this is what turns them back into something RDKit or PyMOL can
draw. Where a real bond graph is available -- reconstruction, where the input
molecule is known -- prefer carrying it through instead of re-perceiving it.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

# Cordero covalent radii (A); covers every symbol in LIGAND_ELEMENT_VOCAB
# except the OTHER catch-all, which gets no bond entries.
_COVALENT_RADII: dict[str, float] = {
    "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "F": 0.57,
    "Cl": 1.02, "Br": 1.20, "I": 1.39, "P": 1.07, "B": 0.84,
    "Si": 1.11,
}
_BOND_TOLERANCE = 0.4
_MIN_ATOMS_FOR_BOND = 2


def read_heavy_atoms(pdb_path: str | Path) -> tuple[list[str], np.ndarray]:
    """Heavy-atom elements and coordinates from a receptor PDB.

    Every ``ATOM``/``HETATM`` record except hydrogens, so cofactors, metals and
    ordered waters count as part of the wall a ligand must not walk through --
    which is what the clash metrics compare against.
    """
    elements: list[str] = []
    coords: list[tuple[float, float, float]] = []
    for line in Path(pdb_path).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        element = (line[76:78].strip() or line[12:16].strip()[:1]).capitalize()
        if element == "H":
            continue
        try:
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except ValueError:
            continue
        elements.append(element)
    return elements, np.asarray(coords, dtype=np.float64).reshape(-1, 3)


def covalent_radius(element: str) -> float:
    """Cordero covalent radius in A; 0.0 for anything not tabulated."""
    return _COVALENT_RADII.get(element, 0.0)


def infer_bonds(
    elements: list[str],
    coords: np.ndarray,
    tol: float = _BOND_TOLERANCE,
) -> list[tuple[int, int]]:
    """Distance-based bond perception: i<j bonded iff d < r_i + r_j + tol."""
    n = len(elements)
    if n < _MIN_ATOMS_FOR_BOND:
        return []
    radii = np.array(
        [_COVALENT_RADII.get(e, 0.0) for e in elements], dtype=np.float64
    )
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    cutoff = radii[:, None] + radii[None, :] + tol
    mask = (dist < cutoff) & (radii[:, None] > 0) & (radii[None, :] > 0)
    mask = np.triu(mask, k=1)
    i_idx, j_idx = np.where(mask)
    return [(int(i), int(j)) for i, j in zip(i_idx, j_idx, strict=True)]


def fmt_atom(  # noqa: PLR0913
    record: str,
    serial: int,
    atom_name: str,
    res_name: str,
    chain_id: str,
    res_seq: int,
    xyz: np.ndarray,
    element: str,
) -> str:
    """Format a single PDB ATOM/HETATM line per the column spec."""
    max_short_name = 4
    name_field = (
        f" {atom_name:<3s}" if len(atom_name) < max_short_name else atom_name[:4]
    )
    return (
        f"{record:<6s}{serial:>5d} {name_field}"
        f" {res_name:>3s} {chain_id:1s}{res_seq:>4d}    "
        f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}"
        f"{1.00:>6.2f}{0.00:>6.2f}          {element:>2s}\n"
    )


def conect_lines(bonds: list[tuple[int, int]], *, start_serial: int) -> list[str]:
    """Emit CONECT records so viewers don't have to guess bonds from distance."""
    # PDB CONECT is 1-indexed by atom serial.
    return [
        f"CONECT{start_serial + a:>5d}{start_serial + b:>5d}\n" for a, b in bonds
    ]


def ligand_lines(
    ligand_elements: list[str],
    ligand_coords: np.ndarray,
    *,
    start_serial: int,
    bonds: list[tuple[int, int]] | None = None,
) -> list[str]:
    """Format the ligand heavy atoms as HETATM (+ optional CONECT) records."""
    lines: list[str] = []
    serial = start_serial
    for k, (elem, xyz) in enumerate(zip(ligand_elements, ligand_coords, strict=True)):
        atom_name = f"{elem}{k + 1}"[:4]
        lines.append(fmt_atom("HETATM", serial, atom_name, "LIG", "L", 1, xyz, elem))
        serial += 1
    if bonds:
        lines.extend(conect_lines(bonds, start_serial=start_serial))
    return lines


def write_full_protein_pdb(
    out_path: Path,
    receptor_pdb_path: Path,
    ligand_elements: list[str],
    ligand_coords: np.ndarray,
    ligand_bonds: list[tuple[int, int]] | None = None,
) -> None:
    """Copy the source receptor PDB verbatim and append the ligand HETATMs."""
    raw = receptor_pdb_path.read_text().splitlines(keepends=True)
    # Drop trailing END/ENDMDL so we can append ligand records before it.
    keep = [ln for ln in raw if ln[:6].strip() not in {"END", "ENDMDL", "MASTER"}]

    last_serial = 0
    for line in keep:
        if line.startswith(("ATOM", "HETATM")):
            with contextlib.suppress(ValueError):
                last_serial = max(last_serial, int(line[6:11]))
    keep.append("TER\n")
    keep.extend(
        ligand_lines(
            ligand_elements,
            ligand_coords,
            start_serial=last_serial + 1,
            bonds=ligand_bonds,
        )
    )
    keep.append("END\n")
    out_path.write_text("".join(keep))


def coord_only_descriptor(
    coord_norm: torch.Tensor,
    full_dim: int,
    coord_field_start: int,
    coord_field_length: int,
) -> np.ndarray:
    """Build a ``(N, full_dim)`` descriptor with only the coord slot populated.

    The ``descriptor_to_coords`` helpers slice out the coord field, so the rest
    of the descriptor can be zeros whenever the caller only wants geometry back
    and does not care about the reconstructed chemistry channels.
    """
    n = coord_norm.shape[0]
    desc = np.zeros((n, full_dim), dtype=np.float32)
    desc[:, coord_field_start : coord_field_start + coord_field_length] = (
        coord_norm.cpu().numpy().astype(np.float32)
    )
    return desc

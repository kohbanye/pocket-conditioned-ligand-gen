"""Structure I/O: read protein backbones and ligands, write reconstruction PDBs.

Uses biotite for PDB parsing (the same library ESM3 relies on) and RDKit for
ligand SDF/MOL. Kept independent of any model package so datasets and metrics
can run without a GPU env.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}
_BACKBONE = ("N", "CA", "C")


@dataclass
class Backbone:
    """Per-residue protein backbone."""

    coords: np.ndarray  # (L, 3, 3) ordered N, CA, C
    seq: str  # length L, 1-letter (X for non-standard)
    res_ids: np.ndarray  # (L,) author residue numbers
    chain_ids: np.ndarray  # (L,) chain id per residue

    @property
    def ca(self) -> np.ndarray:
        return self.coords[:, 1, :]

    def __len__(self) -> int:
        return self.coords.shape[0]


def read_backbone(pdb_path: str | Path, chain: str | None = None) -> Backbone:
    """Read the N/CA/C backbone of (a chain of) a protein PDB.

    Only residues that have all three backbone atoms are kept, in file order.
    """
    import biotite.structure as bs
    import biotite.structure.io.pdbx as pdbx
    from biotite.structure.io.pdb import PDBFile

    pdb_path = Path(pdb_path)
    if pdb_path.suffix in (".cif", ".bcif", ".mmcif"):
        f = pdbx.CIFFile.read(str(pdb_path))
        atoms = pdbx.get_structure(f, model=1)
    else:
        atoms = PDBFile.read(str(pdb_path)).get_structure(model=1)

    atoms = atoms[~atoms.hetero]
    if chain is not None:
        atoms = atoms[atoms.chain_id == chain]
    atoms = atoms[np.isin(atoms.atom_name, _BACKBONE)]

    coords_list: list[np.ndarray] = []
    seq_chars: list[str] = []
    res_ids: list[int] = []
    chain_ids: list[str] = []
    for _, res in _iter_residues(atoms, bs):
        by_name = dict(zip(res.atom_name, res.coord, strict=False))
        if "CA" not in by_name:  # CA is the only hard requirement
            continue
        ca = by_name["CA"]
        coords_list.append(
            np.stack([by_name.get("N", ca), ca, by_name.get("C", ca)])
        )
        resname = str(res.res_name[0])
        seq_chars.append(_THREE_TO_ONE.get(resname, "X"))
        res_ids.append(int(res.res_id[0]))
        chain_ids.append(str(res.chain_id[0]))

    if not coords_list:
        raise ValueError(f"no CA backbone atoms found in {pdb_path}")
    return Backbone(
        coords=np.stack(coords_list).astype(np.float64),
        seq="".join(seq_chars),
        res_ids=np.asarray(res_ids),
        chain_ids=np.asarray(chain_ids),
    )


def _iter_residues(atoms, bs):
    """Yield (residue_starts_index, residue AtomArray) in file order."""
    starts = bs.get_residue_starts(atoms, add_exclusive_stop=True)
    for i in range(len(starts) - 1):
        yield i, atoms[starts[i] : starts[i + 1]]


def read_ligand_heavy(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Read ligand heavy-atom elements + coords from an SDF/MOL/PDB file."""
    path = Path(path)
    if path.suffix.lower() in (".sdf", ".mol", ".mol2"):
        return _read_ligand_rdkit(path)
    return read_hetatm(path)


def _read_ligand_rdkit(path: Path) -> tuple[list[str], np.ndarray]:
    from rdkit import Chem

    if path.suffix.lower() == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=True, sanitize=False)
    else:
        supplier = Chem.SDMolSupplier(str(path), removeHs=True, sanitize=False)
        mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        raise ValueError(f"RDKit could not parse ligand {path}")
    conf = mol.GetConformer()
    elements, coords = [], []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "H":
            continue
        elements.append(atom.GetSymbol())
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append((p.x, p.y, p.z))
    return elements, np.asarray(coords, dtype=np.float64)


def read_hetatm(pdb_path: str | Path) -> tuple[list[str], np.ndarray]:
    """Read HETATM heavy atoms (e.g. the ligand embedded in a recon PDB)."""
    elements, coords = [], []
    for line in Path(pdb_path).read_text().splitlines():
        if not line.startswith("HETATM"):
            continue
        elem = line[76:78].strip() or line[12:16].strip()[0]
        if elem.upper() == "H":
            continue
        elements.append(elem)
        coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return elements, np.asarray(coords, dtype=np.float64).reshape(-1, 3)


def write_backbone_pdb(
    backbone: np.ndarray, seq: str, out_path: str | Path, chain: str = "A"
) -> None:
    """Write a backbone (CA-only or N/CA/C) array to a minimal PDB.

    ``backbone`` is (L, 3) for CA-only or (L, 3, 3) for N/CA/C ordered atoms.
    """
    backbone = np.asarray(backbone, dtype=np.float64)
    one_to_three = {v: k for k, v in _THREE_TO_ONE.items()}
    atom_names = ("CA",) if backbone.ndim == 2 else _BACKBONE
    if backbone.ndim == 2:
        backbone = backbone[:, None, :]

    lines: list[str] = []
    serial = 1
    for i in range(backbone.shape[0]):
        resname = one_to_three.get(seq[i] if i < len(seq) else "G", "GLY")
        for j, name in enumerate(atom_names):
            x, y, z = backbone[i, j]
            # Fixed-width PDB ATOM record (cols: 18-20 resName, 23-26 resSeq,
            # 31-54 xyz, 77-78 element).
            lines.append(
                f"ATOM  {serial:>5} {name:<4} {resname:>3} {chain}{i + 1:>4}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          "
                f"{name[0]:>2}"
            )
            serial += 1
    lines.append("TER")
    lines.append("END")
    Path(out_path).write_text("\n".join(lines) + "\n")

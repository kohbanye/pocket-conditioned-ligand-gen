"""Read SYBYL MOL2 files, which is how CASF ships its docking decoys.

RDKit's MOL2 reader rejects blocks it does not like, and a rejected pose that is
silently skipped is worse than a crash: the target simply scores on fewer
candidates. So this falls back to reading the ATOM/BOND records directly, which
is enough for everything downstream needs (elements, coordinates, bond orders).

Lived in an eval script and was copied -- without the fallback -- into a
benchmark, which meant the two disagreed about how many poses a target has.
"""

from __future__ import annotations

from typing import Any

#: SYBYL bond type -> the integer order the descriptors expect.
_MOL2_BOND = {"1": 1, "2": 2, "3": 3, "ar": 4, "am": 1}
_MIN_ATOM_FIELDS = 6
_MIN_BOND_FIELDS = 4


def mol_to_dict(mol: Any) -> dict | None:  # noqa: ANN401
    """RDKit mol with a conformer -> ``{"atoms": [...], "bonds": [...]}``."""
    from rdkit import Chem  # noqa: PLC0415

    try:
        conf = mol.GetConformer()
    except ValueError:
        return None
    orders = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
    }
    atoms = []
    for i, atom in enumerate(mol.GetAtoms()):
        p = conf.GetAtomPosition(i)
        atoms.append((atom.GetSymbol(), p.x, p.y, p.z))
    bonds = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), orders.get(b.GetBondType(), 1))
        for b in mol.GetBonds()
    ]
    return {"atoms": atoms, "bonds": bonds}


def mol2_records(block: str) -> dict | None:  # noqa: C901
    """Read ATOM/BOND records straight out of a MOL2 block.

    The fallback for blocks RDKit refuses -- e.g. the peptide ligand of CASF
    target 3uri, whose near-native poses carry extra ``NORMAL`` / ``ALT_TYPE``
    sections. Without this those poses vanish from the scored set; for 3uri that
    was every pose under 2 A, which made the target unwinnable and looked like a
    model failure rather than a parsing one.
    """
    if "@<TRIPOS>ATOM" not in block:
        return None
    atoms: list[tuple[str, float, float, float]] = []
    for line in block.split("@<TRIPOS>ATOM")[1].split("@<TRIPOS>")[0].splitlines():
        cols = line.split()
        if len(cols) < _MIN_ATOM_FIELDS:
            continue
        try:
            x, y, z = float(cols[2]), float(cols[3]), float(cols[4])
        except ValueError:
            continue
        # SYBYL type "C.3" / "N.ar" -> element; "Du"/"LP" dummies are skipped.
        element = cols[5].split(".")[0]
        if element in ("Du", "LP"):
            continue
        atoms.append((element.capitalize(), x, y, z))
    if not atoms:
        return None
    bonds: list[tuple[int, int, int]] = []
    if "@<TRIPOS>BOND" in block:
        for line in block.split("@<TRIPOS>BOND")[1].split("@<TRIPOS>")[0].splitlines():
            cols = line.split()
            if len(cols) < _MIN_BOND_FIELDS:
                continue
            try:
                i, j = int(cols[1]) - 1, int(cols[2]) - 1
            except ValueError:
                continue
            if 0 <= i < len(atoms) and 0 <= j < len(atoms):
                bonds.append((i, j, _MOL2_BOND.get(cols[3], 1)))
    return {"atoms": atoms, "bonds": bonds}


def parse_mol2_multi(text: str) -> list[tuple[str, dict]]:
    """``(pose_name, mol_dict)`` per molecule in a multi-``@<TRIPOS>MOLECULE`` file."""
    from rdkit import Chem  # noqa: PLC0415

    out: list[tuple[str, dict]] = []
    for chunk in text.split("@<TRIPOS>MOLECULE")[1:]:
        block = "@<TRIPOS>MOLECULE" + chunk
        name = next((ln.strip() for ln in chunk.splitlines() if ln.strip()), "")
        mol = Chem.MolFromMol2Block(block, sanitize=False, removeHs=False)
        parsed = mol_to_dict(mol) if mol is not None else None
        if parsed is None:
            parsed = mol2_records(block)  # RDKit refused this block
        if parsed is not None:
            out.append((name, parsed))
    return out

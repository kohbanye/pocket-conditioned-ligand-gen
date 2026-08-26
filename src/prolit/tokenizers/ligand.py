"""Ligand 3D structure tokenization with spherical coords + atom features.

Each heavy atom gets a single 30-D descriptor row whose layout is defined in
:mod:`prolit.tokenizers.descriptor_schema`. The descriptor combines:

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
from typing import TYPE_CHECKING

import numpy as np

from prolit.tokenizers.descriptor_schema import (
    K_NEIGHBORS,
    LIGAND_CHARGE_TO_IDX,
    LIGAND_CHARGE_VOCAB,
    LIGAND_ELEMENT_TO_IDX,
    LIGAND_HYBRID_OTHER_IDX,
    LIGAND_NUMH_VOCAB,
    LIGAND_OTHER_IDX,
    LIGAND_RING_NONE_IDX,
)
from prolit.tokenizers.geometry import (
    cartesian_to_spherical,
)

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


def _counts_line(line: str) -> tuple[int, int] | tuple[None, None]:
    """Atom and bond counts off a V2000 counts line.

    The two fields are fixed-width (columns 1-3 and 4-6), so ``split()`` merges
    them the moment either one fills its three digits: " 98101" is 98 atoms and
    101 bonds, not 98101 of anything. Reading it that way makes the parser walk
    the bond block as if it were atoms, which is silent -- the caller gets a
    molecule with three times too many atoms at plausible coordinates. Writers
    that ignore the column widths fall back to ``split()``.
    """
    if len(line) >= 6:  # noqa: PLR2004
        try:
            return int(line[0:3]), int(line[3:6])
        except ValueError:
            pass
    parts = line.split()
    if len(parts) >= 2:  # noqa: PLR2004
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return None, None


def parse_sdf_text(text: str) -> list[dict]:  # noqa: C901
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

        # Counts line (fixed-width; see _counts_line)
        num_atoms, num_bonds = _counts_line(lines[i])
        i += 1
        if num_atoms is None or num_bonds is None:
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


def parse_ligand_pdb_text(
    text: str,
    template_smiles: str | None = None,
    max_template_atoms: int = 70,
) -> dict | None:
    """Parse a ligand PDB block (e.g. BioLIP ``ligand/*.pdb``) into a mol dict.

    BioLIP ligands ship as PDB with coordinates + elements but no reliable bond
    orders. Connectivity is recovered by RDKit proximity bonding; bond ORDERS
    (needed for the aromatic/hybridisation/ring atom features) are then fixed
    from the CCD template SMILES when one is supplied (``ligand.tsv`` gives one
    or more ``;``-separated SMILES per 3-letter CCD id). Falls back to the
    proximity graph (single bonds) if no template matches.

    Returns the same ``{"atoms": [(elem, x, y, z)], "bonds": [(i, j, type)]}``
    shape as :func:`parse_sdf_text` (0-indexed, atom order preserved), or
    ``None`` on parse failure.
    """
    import contextlib  # noqa: PLC0415

    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    mol = Chem.MolFromPDBBlock(
        text, sanitize=False, removeHs=False, proximityBonding=True
    )
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    with contextlib.suppress(Exception):
        mol.UpdatePropertyCache(strict=False)

    # AssignBondOrdersFromTemplate can hang on large / highly-symmetric molecules
    # (combinatorial substructure matching). Such ligands are cofactor-sized and
    # dropped by the downstream heavy-atom filter anyway, so skip the template for
    # them and keep the cheap proximity graph.
    if template_smiles and mol.GetNumHeavyAtoms() <= max_template_atoms:
        for smi in template_smiles.split(";"):
            tmpl = Chem.MolFromSmiles(smi.strip())
            if tmpl is None:
                continue
            try:
                mol = AllChem.AssignBondOrdersFromTemplate(tmpl, mol)
                break
            except Exception:  # noqa: BLE001, S112
                continue

    try:
        conf = mol.GetConformer()
    except ValueError:
        return None

    bond_type_map = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
    }
    atoms = []
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        atoms.append((atom.GetSymbol(), pos.x, pos.y, pos.z))
    bonds = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_type_map.get(b.GetBondType(), 1))
        for b in mol.GetBonds()
    ]
    if not atoms:
        return None
    return {"atoms": atoms, "bonds": bonds}


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


def _atom_features_from_mol(mol: Mol | None, num_atoms: int) -> dict[str, np.ndarray]:
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
    context_coords: np.ndarray | None = None,
    context_elements: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """For each atom, return spherical offsets + element idx of K nearest atoms.

    Distances are computed in the canonical frame; offsets are spherical
    ``(Δr, θ, sin φ, cos φ)`` of the displacement vector ``neighbour - self``.
    Self is excluded. When fewer than ``k`` other atoms exist, the trailing
    slots are zero-padded (offsets) and filled with ``OTHER`` (elements).

    ``context_coords`` extends the SEARCH set without extending the query set:
    neighbours may then be drawn from atoms outside the molecule (in practice
    the pocket), while one row is still emitted per query atom.

    Why this matters. Measured on 2328 generated ligands, the atoms that clash
    with the receptor sit at relative radius 0.716 (all atoms: 0.585) and are
    terminal 39.8% of the time (all atoms: 24.8%) -- the molecule's core fits
    and its substituents poke through the pocket wall. A terminal atom has one
    bonded neighbour, so three of its four knn slots are zero-padded. Letting
    the search set include pocket atoms fills exactly those empty slots, on
    exactly the atoms that clash, at no cost in descriptor width.
    """
    n = canonical_coords.shape[0]
    offsets = np.zeros((n, k * 4), dtype=np.float32)
    nbr_elements = np.full((n, k), LIGAND_OTHER_IDX, dtype=np.int64)

    use_context = context_coords is not None and len(context_coords) > 0
    if use_context:
        # RESERVE the trailing slots for the context rather than merging the two
        # search sets. Merging does nothing: a ligand's own atoms sit at 1.4 A
        # (bonded) and ~2.4 A (next-nearest), while the closest protein atom is
        # 2.7 A away even in a crystal structure, so a plain k-nearest search
        # returns ligand atoms every time and the descriptor came out
        # byte-identical to the context-free one (measured).
        #
        # Splitting costs no bonded information. Slots 1-2 hold the bonded
        # neighbours (median 1.38 / 1.44 A); slots 3-4 are already non-bonded
        # (2.35 / 2.40 A), so handing those two to the pocket replaces
        # second-shell intramolecular geometry -- which the coord block and the
        # first two slots already pin down -- with the receptor contacts the
        # atom has no other way to see.
        k_self = max(1, k // 2)
        k_ctx = k - k_self
    else:
        k_self, k_ctx = k, 0
    if n == 0 or (canonical_coords.shape[0] <= 1 and not use_context):
        return offsets, nbr_elements

    def _fill(coords, elems, exclude_self, base_slot, want) -> None:  # noqa: ANN001
        """Write the ``want`` nearest atoms of ``coords`` from slot ``base_slot``."""
        if want <= 0 or coords.shape[0] == 0:
            return
        dist = np.linalg.norm(
            canonical_coords[:, None, :] - coords[None, :, :], axis=-1
        )
        if exclude_self:
            dist[np.arange(n), np.arange(n)] = np.inf
        take = min(want, coords.shape[0] - (1 if exclude_self else 0))
        if take <= 0:
            return
        idx = np.argpartition(dist, take - 1, axis=1)[:, :take]
        for i in range(n):
            order = np.argsort(dist[i, idx[i]])
            for slot, j in enumerate(idx[i][order]):
                delta = coords[j] - canonical_coords[i]
                r, theta, sphi, cphi = cartesian_to_spherical(delta)
                pos = base_slot + slot
                offsets[i, pos * 4 : pos * 4 + 4] = (r, theta, sphi, cphi)
                nbr_elements[i, pos] = elems[j]

    _fill(canonical_coords, elements, True, 0, k_self)  # noqa: FBT003
    if use_context:
        _fill(
            np.asarray(context_coords),
            np.asarray(context_elements, dtype=elements.dtype),
            False,  # noqa: FBT003
            k_self,
            k_ctx,
        )
    return offsets, nbr_elements



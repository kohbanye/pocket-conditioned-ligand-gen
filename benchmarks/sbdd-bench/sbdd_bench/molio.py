"""Load generated molecules into a uniform representation for evaluation.

Different generators emit different SDFs: DiffSBDD/TargetDiff/DiffGui write
RDKit-perceived bonds; the in-house model writes heavy-atom mol blocks with
single bonds where the 3D coordinates are the source of truth. To score every
model *identically*, each generated entry is loaded as:

* ``elements`` + ``coords`` — always available, straight from the mol block.
* ``mol`` — a best-effort *sanitized* RDKit molecule used for the 2D/topology
  metrics (QED, SA, Lipinski, diversity, novelty). Obtained by, in order:
  (1) reading the SDF entry directly with sanitization; if that fails,
  (2) re-perceiving bonds from the 3D coordinates with Open Babel (the standard
  SBDD fallback — largest fragment, add Hs, sanitize).

Geometry-based metrics (docking, PoseBusters, clashes) never trust the supplied
bond list: they re-derive everything from ``coords`` so a model cannot look good
by shipping a tidy bond block over a distorted geometry.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sbdd_bench import paths

# Elements we are willing to evaluate / dock. "X"/"*" is the VQ-VAE OTHER
# catch-all and cannot be written to XYZ or docked.
REAL_ELEMENTS = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "H"}


@dataclass
class GenMol:
    """One generated molecule in the uniform representation."""

    idx: int
    elements: list[str]
    coords: np.ndarray  # (N, 3) heavy-atom (+ any H) coordinates as written
    mol: object | None = None  # sanitized rdkit.Chem.Mol or None
    smiles: str | None = None
    tag: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_heavy(self) -> int:
        return sum(1 for e in self.elements if e != "H")

    @property
    def has_unknown_element(self) -> bool:
        return any(e not in REAL_ELEMENTS for e in self.elements)


def _xyz_block(elements: list[str], coords: np.ndarray) -> str:
    body = "\n".join(
        f"{e} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"
        for e, c in zip(elements, np.asarray(coords), strict=True)
    )
    return f"{len(elements)}\n\n{body}\n"


def obabel_mol(
    elements: list[str], coords: np.ndarray, *, obabel: str | None = None, add_h: bool = True
):
    """Re-perceive a sanitized RDKit mol from 3D coords via Open Babel.

    Returns the largest sanitized fragment, or ``None``. ``-h`` fills open
    valences so RDKit does not read them as radicals; the largest fragment is
    kept because distorted generated geometry sometimes perceives into pieces.
    """
    from rdkit import Chem

    syms = [str(e) for e in elements]
    if len(syms) < 2 or any(e not in REAL_ELEMENTS for e in syms):
        return None
    obabel = obabel or paths.OBABEL
    with tempfile.TemporaryDirectory() as tmp:
        xyz, sdf = Path(tmp) / "in.xyz", Path(tmp) / "out.sdf"
        xyz.write_text(_xyz_block(syms, coords))
        cmd = [obabel, str(xyz), "-O", str(sdf)]
        if add_h:
            cmd.append("-h")
        subprocess.run(cmd, check=False, capture_output=True)
        if not sdf.exists():
            return None
        supplier = Chem.SDMolSupplier(str(sdf), sanitize=False, removeHs=False)
        mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        return None
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        largest = max(frags, key=lambda m: m.GetNumAtoms())
        Chem.SanitizeMol(largest)
    except Exception:  # noqa: BLE001
        return None
    return largest


def _elements_coords(mol) -> tuple[list[str], np.ndarray]:
    conf = mol.GetConformer()
    els = [a.GetSymbol() for a in mol.GetAtoms()]
    xyz = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())],
        dtype=np.float64,
    )
    return els, xyz


def load_generated(
    sdf_path: str | Path,
    *,
    obabel: str | None = None,
    reperceive: bool = True,
    limit: int | None = None,
) -> list[GenMol]:
    """Load every entry of a generated SDF into :class:`GenMol` records.

    ``reperceive=True`` (default) falls back to Open Babel bond perception from
    coordinates when the SDF entry will not sanitize, so heavy-atom-only outputs
    are scored on the same footing as richer SDFs.
    """
    from rdkit import Chem

    sdf_path = Path(sdf_path)
    out: list[GenMol] = []
    raw = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    for idx, raw_mol in enumerate(raw):
        if limit is not None and idx >= limit:
            break
        if raw_mol is None or raw_mol.GetNumConformers() == 0:
            out.append(GenMol(idx=idx, elements=[], coords=np.zeros((0, 3))))
            continue
        elements, coords = _elements_coords(raw_mol)
        tag = raw_mol.GetProp("_Name") if raw_mol.HasProp("_Name") else f"gen_{idx}"

        # 1) try the model's own bonds, sanitized.
        mol = None
        try:
            cand = Chem.Mol(raw_mol)
            Chem.SanitizeMol(cand)
            mol = cand
        except Exception:  # noqa: BLE001
            mol = None
        # 2) fall back to coordinate-based perception.
        if mol is None and reperceive:
            mol = obabel_mol(elements, coords, obabel=obabel)

        smiles = None
        if mol is not None:
            try:
                smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
            except Exception:  # noqa: BLE001
                smiles = Chem.MolToSmiles(mol)
        out.append(
            GenMol(idx=idx, elements=elements, coords=coords, mol=mol,
                   smiles=smiles, tag=tag)
        )
    return out


def read_ref_mol(ref_sdf: str | Path):
    """Read the reference (co-crystal) ligand as a sanitized RDKit mol."""
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(ref_sdf), sanitize=True, removeHs=True)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:  # last resort: no sanitize
        supplier = Chem.SDMolSupplier(str(ref_sdf), sanitize=False, removeHs=True)
        mol = next((m for m in supplier if m is not None), None)
    return mol

"""Pose-quality metrics (category 3) — the part SBDD models most often fail.

A molecule can have a fine 2D graph and a good docking score yet sit in the
pocket as a physically impossible 3D pose. Three complementary checks:

* **PoseBusters PB-validity** — does the generated conformation pass the
  intramolecular geometry battery (bond lengths, bond angles, ring flatness,
  internal steric clash, sanitization)? This is the DiffGui Table-2 metric. We
  drop ``energy_ratio`` (slow conformer embedding) and ``check_radicals``
  (open valences from heavy-atom reconstruction read as radicals for ~every
  molecule, real GT included — an artifact, not a pose defect).
* **Protein–ligand clashes** — count heavy-atom pairs whose separation is below
  a fraction of their summed van-der-Waals radii. Direct measure of the ligand
  crashing into the receptor (PoseBusters' protein-clash idea, computed here so
  no protein conformer embedding is needed).
* **Strain energy** — MMFF energy of the generated conformer minus the energy
  after relaxation (PoseCheck-style). A large strain means the model placed
  atoms in a high-energy, unnatural geometry.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from sbdd_bench import paths

# van-der-Waals radii (Å), Bondi 1964 + common extras.
_VDW = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "F": 1.47, "Cl": 1.75, "Br": 1.85, "I": 1.98, "B": 1.92, "Si": 2.10,
}


# --------------------------------------------------------------------------
# Protein–ligand steric clashes
# --------------------------------------------------------------------------
def read_protein_heavy(pdb_path: str | Path) -> tuple[list[str], np.ndarray]:
    """Heavy-atom elements + coords from a receptor PDB (ATOM/HETATM, no H)."""
    elements, coords = [], []
    for ln in Path(pdb_path).read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        elem = (ln[76:78].strip() or ln[12:16].strip()[:1]).capitalize()
        if elem == "H":
            continue
        try:
            coords.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
        except ValueError:
            continue
        elements.append(elem)
    return elements, np.asarray(coords, dtype=np.float64).reshape(-1, 3)


def clash_count(
    lig_elements: list[str],
    lig_coords: np.ndarray,
    prot_elements: list[str],
    prot_coords: np.ndarray,
    *,
    tol: float = 0.75,
) -> int:
    """Number of ligand–protein heavy-atom pairs closer than ``tol``·(r_i+r_j)."""
    from scipy.spatial import cKDTree

    lig = np.asarray(
        [
            (e, c)
            for e, c in zip(lig_elements, lig_coords, strict=True)
            if e != "H"
        ],
        dtype=object,
    )
    if len(lig) == 0 or len(prot_coords) == 0:
        return 0
    lig_xyz = np.vstack([c for _, c in lig]).astype(np.float64)
    lig_r = np.array([_VDW.get(str(e), 1.7) for e, _ in lig])
    prot_r = np.array([_VDW.get(str(e), 1.7) for e in prot_elements])
    max_pair = tol * (lig_r.max() + prot_r.max())
    tree = cKDTree(prot_coords)
    n = 0
    for i in range(len(lig_xyz)):
        for j in tree.query_ball_point(lig_xyz[i], max_pair):
            d = np.linalg.norm(lig_xyz[i] - prot_coords[j])
            if d < tol * (lig_r[i] + prot_r[j]):
                n += 1
    return n


# --------------------------------------------------------------------------
# Strain energy (PoseCheck-style)
# --------------------------------------------------------------------------
def strain_energy(mol, *, max_iters: int = 200) -> float | None:
    """E(generated conformer) − E(MMFF-relaxed conformer), kcal/mol.

    A proxy for how distorted the generated pose is. Needs a sanitized mol with
    a 3D conformer and MMFF parameters; returns ``None`` if either is missing.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if mol is None or mol.GetNumConformers() == 0:
        return None
    try:
        m = Chem.AddHs(mol, addCoords=True)
        props = AllChem.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94s")
        if props is None:
            return None
        ff0 = AllChem.MMFFGetMoleculeForceField(m, props)
        e_gen = ff0.CalcEnergy()
        ff1 = AllChem.MMFFGetMoleculeForceField(m, props)
        ff1.Minimize(maxIts=max_iters)
        e_min = ff1.CalcEnergy()
        return float(e_gen - e_min)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# PoseBusters PB-validity (batched)
# --------------------------------------------------------------------------
def _reconstruct_for_pb(gen_mols) -> dict[int, object]:
    """Largest sanitized fragment per generated mol, rebuilt via Open Babel from
    coordinates (with Hs) so PoseBusters sees filled valences."""
    from rdkit import Chem

    from sbdd_bench.molio import REAL_ELEMENTS

    frames = []
    for g in gen_mols:
        syms = [str(e) for e in g.elements]
        if len(syms) < 2 or any(e not in REAL_ELEMENTS for e in syms):
            continue
        body = "\n".join(
            f"{e} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"
            for e, c in zip(syms, np.asarray(g.coords), strict=True)
        )
        frames.append(f"{len(syms)}\n{g.idx}\n{body}\n")
    mols: dict[int, object] = {}
    if not frames:
        return mols
    with tempfile.TemporaryDirectory() as tmp:
        xyz, sdf = Path(tmp) / "in.xyz", Path(tmp) / "out.sdf"
        xyz.write_text("".join(frames))
        subprocess.run([paths.OBABEL, str(xyz), "-O", str(sdf), "-h"],
                       check=False, capture_output=True)
        if not sdf.exists():
            return mols
        for mol in Chem.SDMolSupplier(str(sdf), sanitize=False, removeHs=False):
            if mol is None:
                continue
            try:
                orig = int(mol.GetProp("_Name").strip())
                frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                largest = max(frags, key=lambda m: m.GetNumAtoms())
                Chem.SanitizeMol(largest)
            except Exception:  # noqa: BLE001
                continue
            mols[orig] = largest
    return mols


def pb_validity(gen_mols, *, chunk: int = 1000, max_workers: int = 4) -> dict[int, bool]:
    """PoseBusters PB-validity per generated-mol idx. Idx absent ⇒ not bustable."""
    from posebusters import PoseBusters

    mols_by_idx = _reconstruct_for_pb(gen_mols)
    idxs = sorted(mols_by_idx)
    if not idxs:
        return {}
    base = PoseBusters(config="mol")
    cfg = base.config
    drop = {"energy_ratio", "check_radicals"}
    cfg["modules"] = [m for m in cfg["modules"] if m.get("function") not in drop]
    cfg["max_workers"] = max_workers
    buster = PoseBusters(config=cfg)
    out: dict[int, bool] = {}
    for start in range(0, len(idxs), chunk):
        batch_idx = idxs[start : start + chunk]
        df = buster.bust([mols_by_idx[i] for i in batch_idx])
        passed = df.all(axis=1).to_numpy()
        for j, i in enumerate(batch_idx):
            out[i] = bool(passed[j]) if j < len(passed) else False
    return out

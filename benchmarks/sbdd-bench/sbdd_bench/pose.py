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
    """Heavy-atom elements + coords from a receptor PDB (ATOM/HETATM, no H).

    Delegates to :func:`prolit.api.read_heavy_atoms` so the clash metric here
    and the rigid steric relief in ``scripts/`` read the receptor identically;
    a difference between the two would show up as a fix that does not fix.
    """
    from prolit.api import read_heavy_atoms

    return read_heavy_atoms(pdb_path)


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
    """One sanitized molecule per generated mol, for PoseBusters to bust.

    **The model's own bonds win when it has any.** A model that writes a bond
    block has committed to a chemistry, and that commitment is what the table
    should score -- it is already what ``validity``, the SMILES, QED and SA are
    computed from (``molio.load_generated`` sanitizes it and only falls back).
    Re-perceiving those coordinates with Open Babel instead scored the two
    halves of the table on two different molecules, and the perceived half was
    the worse one: on 808 ProLIT ligands it turned 26% of them into structures
    RDKit could not even convert to InChI (``inchi_convertible`` 0.965 -> 0.704,
    PB-validity 0.589 -> 0.510), typically by handing a nitrogen a fourth bond.

    Perception stays as the fallback, unchanged, for the coordinate-only models
    -- the rule is the same for every model, and a model that emits valid bonds
    is genuinely better at that than one that leaves them to be guessed. It is
    also the convention the published numbers use: MiDi, FLOWR and TargetDiff
    all bust their own predicted bonds.

    Hydrogens are left however each path produced them; PoseBusters' ``mol``
    checks were measured to return bit-identical results with and without them.
    """
    from rdkit import Chem

    from sbdd_bench.molio import REAL_ELEMENTS

    mols: dict[int, object] = {}
    needs_perception = []
    for g in gen_mols:
        if getattr(g, "mol", None) is not None:
            mols[g.idx] = g.mol
        else:
            needs_perception.append(g)

    frames = []
    for g in needs_perception:
        syms = [str(e) for e in g.elements]
        if len(syms) < 2 or any(e not in REAL_ELEMENTS for e in syms):
            continue
        body = "\n".join(
            f"{e} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"
            for e, c in zip(syms, np.asarray(g.coords), strict=True)
        )
        frames.append(f"{len(syms)}\n{g.idx}\n{body}\n")
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


# --- bond-geometry distances against the reference ligands ---------------------
#
# FLOWR and the MiDi line of work report Wasserstein-1 distances between the
# generated molecules' bond-length and bond-angle distributions and the test
# set's, because a model can put every atom in a plausible place and still get
# the local geometry systematically wrong -- and PoseBusters only answers
# pass/fail, not by how much.
#
# Conditioned on bond type (element pair + order) and on the central atom for
# angles, NOT pooled. Pooling would let a shift in composition masquerade as a
# geometry error: a model that draws more C-C and fewer C=O moves the pooled
# histogram without any individual bond being wrong. Per-type distances are then
# averaged weighted by how often the REFERENCE uses each type, so the summary is
# "how wrong is a typical bond of the kind real ligands contain".
#
# The absolute value is only comparable across papers when their conditioning
# matches; against arms measured here it is comparable by construction.


def _bond_key(bond) -> str:
    a, b = bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()
    return f"{min(a, b)}-{max(a, b)}:{bond.GetBondTypeAsDouble():g}"


def bond_geometry(mol) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Bond lengths and bond angles of one molecule, keyed by type."""
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    lengths: dict[str, list[float]] = {}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        lengths.setdefault(_bond_key(bond), []).append(
            float(np.linalg.norm(pos[i] - pos[j]))
        )
    angles: dict[str, list[float]] = {}
    for atom in mol.GetAtoms():
        nb = [n.GetIdx() for n in atom.GetNeighbors()]
        c = atom.GetIdx()
        for x in range(len(nb)):
            for y in range(x + 1, len(nb)):
                u = pos[nb[x]] - pos[c]
                v = pos[nb[y]] - pos[c]
                nu, nv = np.linalg.norm(u), np.linalg.norm(v)
                if nu < 1e-6 or nv < 1e-6:
                    continue
                cos = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
                angles.setdefault(atom.GetSymbol(), []).append(
                    float(np.degrees(np.arccos(cos)))
                )
    return lengths, angles


def _w1(a: list[float], b: list[float]) -> float:
    """Wasserstein-1 between two samples, by quantile matching."""
    if not a or not b:
        return float("nan")
    n = max(len(a), len(b), 64)
    q = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def bond_w1(gen_mols, ref_mols) -> dict[str, float]:
    """Reference-frequency-weighted W1 for bond lengths (A) and angles (deg).

    ``nan`` when the reference supplies no bonds at all, which is a missing
    reference rather than a perfect model, and must not average in as zero.
    """
    def collect(mols):
        L: dict[str, list[float]] = {}
        A: dict[str, list[float]] = {}
        for m in mols:
            if m is None or m.GetNumConformers() == 0:
                continue
            ll, aa = bond_geometry(m)
            for k, v in ll.items():
                L.setdefault(k, []).extend(v)
            for k, v in aa.items():
                A.setdefault(k, []).extend(v)
        return L, A

    gl, ga = collect(gen_mols)
    rl, ra = collect(ref_mols)

    def weighted(gen: dict, ref: dict) -> float:
        total = sum(len(v) for v in ref.values())
        if not total:
            return float("nan")
        acc = 0.0
        for key, rv in ref.items():
            gv = gen.get(key)
            if not gv:
                continue  # the model never drew this type; absence is a
                # composition error, which the length/angle metric does not score
            acc += (len(rv) / total) * _w1(gv, rv)
        return acc

    return {"bond_length_w1": weighted(gl, rl), "bond_angle_w1": weighted(ga, ra)}

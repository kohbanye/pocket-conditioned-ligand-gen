"""Pocket-aware local relaxation of generated ligand poses (physics only, no Vina).

``vina_score`` is scored on the pose **as generated**, so unlike ``vina_dock`` it is
sensitive to exactly the thing the pose refiner is meant to fix. On the 100-pocket
set our arms sit at ``vina_score ~ -2.9`` while ``vina_min`` (Vina's own local
optimisation of the same pose) reaches ``-5.4``: ~2.5 kcal/mol of the gap to the
target is pure local slack -- intramolecular strain plus ligand-pocket overlap --
that a local optimiser recovers without changing the molecule.

This module closes that slack with a physics objective that never touches Vina's
scoring function:

    E = E_UFF(ligand)                       intramolecular (RDKit UFF)
      + w_pkt * softLJ(ligand, pocket)      intermolecular, pocket held rigid
      + w_tether * |x - x0|^2               keeps the LM's binding-mode choice

``softLJ`` is a shifted 8-4 well with its minimum at the pair's van der Waals
contact distance, so it removes overlap *without* pushing the ligand out of the
site -- a purely repulsive term would evacuate the pocket and lose the attractive
contacts Vina rewards. The pocket is rigid, only ligand heavy atoms move, and the
tether bounds how far the pose may drift from the generated one.

Usage (per target, so a crash cannot take out the whole run)::

    python scripts/relax_in_pocket.py --arm sep4096_cs --out-arm sep4096_cs_rx \
        --targets ABL2_HUMAN_274_551_0 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.optimize import minimize

SBDD_BENCH = Path("/gs/bs/tga-ohuelab/sakano/git/sbdd-bench")
RDLogger.DisableLog("rdApp.*")

# Bondi van der Waals radii (A) for the elements the tokenizer can emit.
_VDW: dict[str, float] = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "F": 1.47,
    "Cl": 1.75, "Br": 1.85, "I": 1.98, "P": 1.80, "B": 1.92,
    "Si": 2.10, "H": 1.20,
}
_DEFAULT_VDW = 1.70


def load_pocket(receptor_pdb: Path) -> tuple[np.ndarray, np.ndarray]:
    """Heavy-atom coordinates + vdW radii of the receptor (ATOM/HETATM records)."""
    xyz, rad = [], []
    for line in receptor_pdb.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        elem = (line[76:78].strip() or line[12:16].strip()[:1]).capitalize()
        if elem == "H":
            continue
        try:
            xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except ValueError:
            continue
        rad.append(_VDW.get(elem, _DEFAULT_VDW))
    return np.asarray(xyz, dtype=np.float64), np.asarray(rad, dtype=np.float64)


def _repulsion(
    x: np.ndarray,
    pkt: np.ndarray,
    d0: np.ndarray,
) -> tuple[float, np.ndarray]:
    """One-sided harmonic overlap penalty: e = sum max(0, d0 - d)^2.

    Purely repulsive by design. An attractive well (the 8-4 form tried first)
    maximises contact instead of relieving overlap: measured on 12 poses it took
    ligand-pocket pairs under 3 A from 3.5 to 21.8 per ligand and pushed
    ``vina_score`` from -3.9 to +1.9. Vina's own attractive term is already
    satisfied by the LM's placement, so the only thing to fix here is overlap.
    """
    diff = x[:, None, :] - pkt[None, :, :]
    dist = np.sqrt((diff**2).sum(-1) + 1e-12)
    over = d0 - dist
    mask = over > 0.0
    if not mask.any():
        return 0.0, np.zeros_like(x)
    energy = float((np.where(mask, over, 0.0) ** 2).sum())
    # de/dd = -2 (d0 - d) for overlapping pairs
    dedd = np.where(mask, -2.0 * over, 0.0)
    grad = (dedd[:, :, None] * diff / dist[:, :, None]).sum(axis=1)
    return energy, grad


def _soft_lj(
    x: np.ndarray,
    pkt: np.ndarray,
    d0: np.ndarray,
    cutoff: float,
) -> tuple[float, np.ndarray]:
    """Shifted 8-4 well, minimum at ``d0`` (kept for the ablation; see above)."""
    diff = x[:, None, :] - pkt[None, :, :]
    dist = np.sqrt((diff**2).sum(-1) + 1e-12)
    mask = dist < cutoff
    if not mask.any():
        return 0.0, np.zeros_like(x)
    ratio = np.where(mask, d0 / dist, 0.0)
    r4 = ratio**4
    r8 = r4 * r4
    energy = float(np.where(mask, r8 - 2.0 * r4, 0.0).sum())
    # de/dd = (-8 r8 + 8 r4) / d
    dedd = np.where(mask, (-8.0 * r8 + 8.0 * r4) / dist, 0.0)
    grad = (dedd[:, :, None] * diff / dist[:, :, None]).sum(axis=1)
    return energy, grad


def _internal_restraint(
    x: np.ndarray,
    pairs: np.ndarray,
    ref: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Harmonic restraint holding 1-2/1-3/1-4 distances at their generated values.

    Relieving pocket overlap atom-by-atom with no intramolecular term destroys the
    molecule: measured on the 100-pocket set it took PoseBusters validity from 0.44
    to 0.11 and tripled the strain, which is why the raw ``vina_score`` looked so
    good -- the atoms had simply been spread apart. A full UFF term overcorrects the
    other way (it relaxes toward a vacuum geometry and loses the pocket fit), so
    what is restrained here is the molecule's OWN bonded geometry: fixing the 1-2,
    1-3 and 1-4 distances pins bond lengths, bond angles and ring shape while
    leaving the rigid-body and torsional freedom that the overlap relief needs.
    """
    if pairs.shape[0] == 0:
        return 0.0, np.zeros_like(x)
    diff = x[pairs[:, 0]] - x[pairs[:, 1]]
    dist = np.sqrt((diff**2).sum(-1) + 1e-12)
    dev = dist - ref
    energy = float((dev**2).sum())
    g = (2.0 * dev / dist)[:, None] * diff
    grad = np.zeros_like(x)
    np.add.at(grad, pairs[:, 0], g)
    np.add.at(grad, pairs[:, 1], -g)
    return energy, grad


def _bonded_pairs(mol: Chem.Mol, max_path: int) -> np.ndarray:
    """All atom pairs separated by at most ``max_path`` bonds (1-2 .. 1-max_path)."""
    n = mol.GetNumAtoms()
    dm = Chem.GetDistanceMatrix(mol)
    idx = [(i, j) for i in range(n) for j in range(i + 1, n) if 1 <= dm[i, j] <= max_path]
    return np.asarray(idx, dtype=np.int64).reshape(-1, 2)


def _attraction(
    x: np.ndarray,
    pkt: np.ndarray,
    d0: np.ndarray,
    sigma: float,
) -> tuple[float, np.ndarray]:
    """Gaussian contact well centred on the vdW contact distance.

    Vina rewards atoms sitting at contact distance, and the LM's placement does
    not always get there. A bare attractive term was tried first and failed --
    with no intramolecular restraint it simply crushed the ligand onto the pocket
    wall -- but with the 1-2/1-3/1-4 restraint holding the molecule rigid it can
    only translate/rotate/torsion the pose into better contact, which is the
    intended effect.
    """
    diff = x[:, None, :] - pkt[None, :, :]
    dist = np.sqrt((diff**2).sum(-1) + 1e-12)
    dev = dist - d0
    g = np.exp(-0.5 * (dev / sigma) ** 2)
    energy = -float(g.sum())
    dedd = g * dev / (sigma**2)
    grad = (dedd[:, :, None] * diff / dist[:, :, None]).sum(axis=1)
    return energy, grad


def _random_rigid(
    x: np.ndarray, rng: np.random.Generator, trans: float, rot_deg: float
) -> np.ndarray:
    """Random rigid-body displacement about the ligand centroid."""
    if trans <= 0 and rot_deg <= 0:
        return x
    c = x.mean(axis=0)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-12
    ang = np.deg2rad(rng.normal(0.0, rot_deg))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + np.sin(ang) * k + (1 - np.cos(ang)) * (k @ k)
    return (x - c) @ rot.T + c + rng.normal(0.0, trans, size=3)


def relax_mol(  # noqa: PLR0913
    mol: Chem.Mol,
    pkt_xyz: np.ndarray,
    pkt_rad: np.ndarray,
    *,
    w_pkt: float,
    w_tether: float,
    cutoff: float,
    contact_scale: float,
    maxiter: int,
    pkt_mode: str = "repulsive",
    w_uff: float = 0.0,
    w_internal: float = 0.0,
    internal_path: int = 3,
    w_att: float = 0.0,
    att_sigma: float = 0.8,
    multi_start: int = 1,
    start_trans: float = 1.0,
    start_rot: float = 15.0,
    seed: int = 0,
) -> Chem.Mol | None:
    """Relax one ligand pose in the rigid pocket; returns a new conformer or None."""
    try:
        ff = AllChem.UFFGetMoleculeForceField(mol)
    except Exception:  # noqa: BLE001
        return None
    if ff is None:
        return None

    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    x0 = np.asarray(conf.GetPositions(), dtype=np.float64)

    lig_rad = np.array(
        [_VDW.get(a.GetSymbol(), _DEFAULT_VDW) for a in mol.GetAtoms()],
        dtype=np.float64,
    )
    # Restrict the pocket to atoms that can ever be in range of this ligand.
    near = (
        np.linalg.norm(pkt_xyz[None, :, :] - x0[:, None, :], axis=-1).min(axis=0)
        < cutoff + 4.0
    )
    pkt = pkt_xyz[near]
    if pkt.shape[0] == 0:
        return None
    d0 = contact_scale * (lig_rad[:, None] + pkt_rad[near][None, :])

    if w_internal > 0.0:
        int_pairs = _bonded_pairs(mol, internal_path)
        int_ref = (
            np.linalg.norm(x0[int_pairs[:, 0]] - x0[int_pairs[:, 1]], axis=1)
            if int_pairs.shape[0]
            else np.zeros(0)
        )
    else:
        int_pairs = np.zeros((0, 2), dtype=np.int64)
        int_ref = np.zeros(0)

    def fun(flat: np.ndarray) -> tuple[float, np.ndarray]:
        x = flat.reshape(n, 3)
        if w_uff > 0.0:
            e_uff = w_uff * ff.CalcEnergy(list(flat))
            g_uff = w_uff * np.asarray(ff.CalcGrad(list(flat)), dtype=np.float64)
        else:
            e_uff = 0.0
            g_uff = np.zeros(n * 3, dtype=np.float64)
        if pkt_mode == "repulsive":
            e_lj, g_lj = _repulsion(x, pkt, d0)
        else:
            e_lj, g_lj = _soft_lj(x, pkt, d0, cutoff)
        d = x - x0
        e_t = float((d**2).sum())
        g_t = 2.0 * d
        e_in, g_in = _internal_restraint(x, int_pairs, int_ref)
        energy = e_uff + w_pkt * e_lj + w_tether * e_t + w_internal * e_in
        grad = g_uff + (w_pkt * g_lj + w_tether * g_t + w_internal * g_in).ravel()
        if w_att > 0.0:
            e_a, g_a = _attraction(x, pkt, d0, att_sigma)
            energy = energy + w_att * e_a
            grad = grad + (w_att * g_a).ravel()
        return energy, grad

    # Multi-start: the objective is non-convex in the ligand's rigid-body pose, so
    # a few randomly displaced starts explore alternative placements. The winner is
    # chosen by this physics objective alone -- Vina is never consulted.
    rng = np.random.default_rng(seed)
    best_x, best_e = None, np.inf
    for k in range(max(1, multi_start)):
        start = x0 if k == 0 else _random_rigid(x0, rng, start_trans, start_rot)
        res = minimize(
            fun, start.ravel(), jac=True, method="L-BFGS-B",
            options={"maxiter": maxiter},
        )
        if res.fun < best_e:
            best_e, best_x = float(res.fun), res.x
    out = Chem.Mol(mol)
    oc = out.GetConformer()
    for i, (px, py, pz) in enumerate(best_x.reshape(n, 3)):
        oc.SetAtomPosition(i, (float(px), float(py), float(pz)))
    return out


def main() -> None:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, help="source arm dir under sbdd-bench/outputs")
    ap.add_argument("--out-arm", required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--w-pkt", type=float, default=1.0)
    # Intramolecular UFF weight. Vacuum UFF relaxation expands the ligand and
    # destroys the pocket complementarity the LM found: on a 9-target subset it
    # took vina_score from -4.62 to -1.43 on its own. Default 0 keeps the pose
    # rigid apart from the overlap relief.
    ap.add_argument("--w-uff", type=float, default=0.0)
    ap.add_argument("--w-tether", type=float, default=0.05)
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--contact-scale", type=float, default=1.0)
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--pkt-mode", choices=("repulsive", "lj"), default="repulsive")
    ap.add_argument("--w-internal", type=float, default=0.0)
    ap.add_argument("--internal-path", type=int, default=3)
    ap.add_argument("--pocket-source", choices=("pocket", "receptor"), default="pocket")
    # Training poses come from a different target index (CrossDocked train
    # pockets) than the evaluation set.
    ap.add_argument(
        "--index", type=Path, default=SBDD_BENCH / "data" / "targets" / "index.json"
    )
    ap.add_argument("--w-att", type=float, default=0.0)
    ap.add_argument("--att-sigma", type=float, default=0.8)
    ap.add_argument("--multi-start", type=int, default=1)
    ap.add_argument("--start-trans", type=float, default=1.0)
    ap.add_argument("--start-rot", type=float, default=15.0)
    args = ap.parse_args()

    index = json.loads(args.index.read_text())
    targets = index["targets"] if isinstance(index, dict) and "targets" in index else index
    by_id = {t["target_id"]: t for t in targets}

    for tid in args.targets:
        meta = by_id.get(tid)
        src = SBDD_BENCH / "outputs" / args.arm / "own" / tid / "generated.sdf"
        if meta is None or not src.exists():
            print(f"[relax] {tid}: missing arm sdf or index entry", flush=True)
            continue
        # Vina scores against the FULL receptor, so relieving overlap against only
        # the extracted pocket (~570 atoms vs ~2100) can leave clashes with atoms
        # just outside it that Vina still charges for. The pocket file is the
        # cheaper default; --pocket-source receptor uses everything.
        key = "pocket_pdb" if args.pocket_source == "pocket" else "receptor_pdb"
        pkt_xyz, pkt_rad = load_pocket(args.index.parent / meta[key])
        if pkt_xyz.shape[0] == 0:
            print(f"[relax] {tid}: empty receptor", flush=True)
            continue

        dst_dir = SBDD_BENCH / "outputs" / args.out_arm / "own" / tid
        dst_dir.mkdir(parents=True, exist_ok=True)
        n_in = n_out = 0
        with Chem.SDWriter(str(dst_dir / "generated.sdf")) as w:
            for mol in Chem.SDMolSupplier(str(src), sanitize=True, removeHs=True):
                if mol is None or mol.GetNumConformers() == 0:
                    continue
                n_in += 1
                out = relax_mol(
                    mol,
                    pkt_xyz,
                    pkt_rad,
                    w_pkt=args.w_pkt,
                    w_tether=args.w_tether,
                    cutoff=args.cutoff,
                    contact_scale=args.contact_scale,
                    maxiter=args.maxiter,
                    pkt_mode=args.pkt_mode,
                    w_uff=args.w_uff,
                    w_internal=args.w_internal,
                    internal_path=args.internal_path,
                    w_att=args.w_att,
                    att_sigma=args.att_sigma,
                    multi_start=args.multi_start,
                    start_trans=args.start_trans,
                    start_rot=args.start_rot,
                )
                w.write(out if out is not None else mol)
                n_out += int(out is not None)
        print(f"[relax] {tid}: {n_in} mols, {n_out} relaxed", flush=True)


if __name__ == "__main__":
    sys.exit(main())

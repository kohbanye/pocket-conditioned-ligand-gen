"""Where does the pose lose its score: clashing, or failing to touch?

Vina's total hides this. Its intermolecular energy is a weighted sum of five
terms and they pull in opposite directions -- ``repulsion`` charges for
overlap, while ``gauss1``/``gauss2``/``hydrophobic``/``hbond`` pay for contact.
A pose 2 A off its site loses on the contact terms without necessarily
clashing; a pose jammed into the wall loses on repulsion while contacting
plenty. The fix is different in each case, so the total is the wrong number to
look at.

The five terms are published and simple functions of the surface distance
``d = r_ij - R_i - R_j``. This implements them and **validates the sum against
the intermolecular energy Vina itself reports**, so the decomposition is not
taken on trust.

Typing is element-based (C/halogen hydrophobic, N/O polar), which is what Vina
does at the pdbqt level minus the donor/acceptor distinction -- so ``hbond``
here is an upper bound on the real term. The validation column shows how much
that costs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

BENCH = Path(__file__).resolve().parent.parent
REPO = BENCH.parent.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(REPO / "src"))

from sbdd_bench import datasets, docking, molio  # noqa: E402

# Vina 1.2 weights (Trott & Olson 2010, as shipped)
W = {"gauss1": -0.035579, "gauss2": -0.005156, "repulsion": 0.840245,
     "hydrophobic": -0.035069, "hbond": -0.587439}
# X-S radii Vina uses
RADII = {"C": 1.9, "N": 1.8, "O": 1.7, "S": 2.0, "P": 2.1, "F": 1.5,
         "Cl": 1.8, "Br": 2.0, "I": 2.2, "H": 1.0}
HYDROPHOBIC = {"C", "F", "Cl", "Br", "I"}
POLAR = {"N", "O"}


def terms(lig_el, lig_xyz, rec_el, rec_xyz, cutoff: float = 8.0) -> dict:
    """Per-term intermolecular energy. Heavy atoms only, as Vina scores them."""
    li = [i for i, e in enumerate(lig_el) if e != "H"]
    ri = [i for i, e in enumerate(rec_el) if e != "H"]
    if not li or not ri:
        return dict.fromkeys(W, 0.0)
    L, R = np.asarray(lig_xyz)[li], np.asarray(rec_xyz)[ri]
    le = [lig_el[i] for i in li]
    re_ = [rec_el[i] for i in ri]
    d = np.linalg.norm(L[:, None, :] - R[None, :, :], axis=2)
    near = d < cutoff
    lr = np.array([RADII.get(e, 1.9) for e in le])
    rr = np.array([RADII.get(e, 1.9) for e in re_])
    surf = d - lr[:, None] - rr[None, :]
    out = {}
    out["gauss1"] = float((np.exp(-((surf / 0.5) ** 2)) * near).sum())
    out["gauss2"] = float((np.exp(-(((surf - 3.0) / 2.0) ** 2)) * near).sum())
    out["repulsion"] = float(((surf < 0) * surf**2 * near).sum())
    hl = np.array([e in HYDROPHOBIC for e in le])
    hr = np.array([e in HYDROPHOBIC for e in re_])
    hmask = hl[:, None] & hr[None, :] & near
    h = np.clip((1.5 - surf) / 1.0, 0.0, 1.0)
    out["hydrophobic"] = float((h * hmask).sum())
    pl = np.array([e in POLAR for e in le])
    pr = np.array([e in POLAR for e in re_])
    pmask = pl[:, None] & pr[None, :] & near
    b = np.clip(-surf / 0.7, 0.0, 1.0)
    out["hbond"] = float((b * pmask).sum())
    return out


def vina_intermolecular(elements, coords, target) -> float | None:
    """The number Vina itself reports, for validation."""
    with tempfile.TemporaryDirectory() as td:
        xyz, pdbqt = Path(td) / "l.xyz", Path(td) / "l.pdbqt"
        docking._write_xyz(xyz, elements, [list(map(float, c)) for c in coords])  # noqa: SLF001
        subprocess.run(  # noqa: S603
            ["/home/5/uq02055/usr/app/babel/bin/obabel", str(xyz), "-O", str(pdbqt),
             "-r", "-p", "7.4", "--partialcharge", "gasteiger"],
            capture_output=True, check=False,
        )
        if not pdbqt.exists():
            return None
        c = np.asarray(coords, dtype=float).mean(0)
        r = subprocess.run(  # noqa: S603
            ["/home/5/uq02055/.local/bin/vina", "--ligand", str(pdbqt),
             "--receptor", str(target.receptor_pdbqt), "--score_only", "--cpu", "1",
             "--center_x", f"{c[0]:.3f}", "--center_y", f"{c[1]:.3f}",
             "--center_z", f"{c[2]:.3f}",
             "--size_x", "25", "--size_y", "25", "--size_z", "25"],
            capture_output=True, text=True, check=False,
        )
        for line in r.stdout.splitlines():
            if "Final Intermolecular Energy" in line:
                try:
                    return float(line.split(":")[1].split("(")[0])
                except (IndexError, ValueError):
                    return None
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tree",
        action="append",
        default=None,
        metavar="LABEL=RELPATH",
        help="generation tree to decompose, relative to benchmarks/sbdd-bench/"
        "outputs (e.g. 'ProLIT=gen_t0.7_ml/own'). Repeatable. These used to be "
        "hard-coded to the post-processed tree, which is not what an ML-only "
        "table reports.",
    )
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--per-target", type=int, default=12)
    p.add_argument("--validate", type=int, default=30,
                   help="how many poses to also score with Vina itself")
    p.add_argument("--shard", default=None)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    from sbdd_bench import pose as bpose

    OUT = BENCH / "outputs"
    TREES = (
        [(spec.split("=", 1)[0], OUT / spec.split("=", 1)[1]) for spec in a.tree]
        if a.tree
        else [
            ("ProLIT", OUT / "gen100_arom_post" / "own"),
            ("FLOWR", OUT / "flowr100" / "flowr"),
        ]
    )
    ts = {x.target_id: x for x in datasets.load_targets()[: a.limit]}
    tids = sorted(set(ts) & set.intersection(
        *[{q.parent.name for q in tr.glob("*/generated.sdf")} for _, tr in TREES]))
    if a.shard:
        k, n = (int(v) for v in a.shard.split("/"))
        tids = tids[k::n]

    rows, n_val = [], 0
    for tid in tids:
        tg = ts[tid]
        rec_el, rec_xyz = bpose.read_protein_heavy(tg.receptor_pdb)
        try:
            ref = next(m for m in molio.load_generated(Path(tg.ref_ligand_sdf))
                       if m.mol is not None)
        except Exception:  # noqa: BLE001
            continue
        srcs = [("reference", [ref])]
        for nm, tree in TREES:
            try:
                srcs.append((nm, [g for g in molio.load_generated(tree / tid / "generated.sdf")
                                  if g.tag != "ref" and g.mol is not None][: a.per_target]))
            except Exception:  # noqa: BLE001
                pass
        for nm, gens in srcs:
            for g in gens:
                el = list(g.elements)
                xyz = np.asarray(g.coords, dtype=float)
                if len(el) != len(xyz) or len(el) < 3:
                    continue
                t = terms(el, xyz, rec_el, rec_xyz)
                e = sum(W[k] * v for k, v in t.items())
                row = {"tid": tid, "src": nm, "n_heavy": int(sum(1 for x in el if x != "H")),
                       "energy": e, **t}
                if n_val < a.validate:
                    v = vina_intermolecular(el, xyz, tg)
                    if v is not None:
                        row["vina_inter"] = v
                        n_val += 1
                rows.append(row)
        print(f"{tid[:28]:28s} {len(rows)} rows", flush=True)
    a.out.write_text(json.dumps(rows))
    print(f"wrote {len(rows)} rows ({n_val} validated)")


if __name__ == "__main__":
    main()

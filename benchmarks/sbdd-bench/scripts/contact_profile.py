"""How close does each ligand atom get to the receptor, and is that right?

The Vina total says a pose is bad. It does not say whether the atoms are
jammed into the wall, hovering in the middle of the cavity, or sticking out
into solvent. Those look identical in the total and need different fixes.

For every ligand heavy atom this records the distance to the nearest receptor
heavy atom, minus the two vdW radii -- the *surface* gap. Vina's optimum is
around 0, clashes are negative, and anything past about +1.5 A contributes
almost nothing. The shape of that distribution, compared against the reference
ligands and against FLOWR, says which failure is happening.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

BENCH = Path(__file__).resolve().parent.parent
REPO = BENCH.parent.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(REPO / "src"))

from sbdd_bench import datasets, molio  # noqa: E402
from sbdd_bench import pose as bpose  # noqa: E402

RADII = {"C": 1.9, "N": 1.8, "O": 1.7, "S": 2.0, "P": 2.1, "F": 1.5,
         "Cl": 1.8, "Br": 2.0, "I": 2.2}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--per-target", type=int, default=20)
    p.add_argument("--shard", default=None)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    OUT = BENCH / "outputs"
    TREES = [("ProLIT", OUT / "gen100_arom_post" / "own"),
             ("FLOWR", OUT / "flowr100" / "flowr")]
    ts = {x.target_id: x for x in datasets.load_targets()[: a.limit]}
    tids = sorted(set(ts) & set.intersection(
        *[{q.parent.name for q in tr.glob("*/generated.sdf")} for _, tr in TREES]))
    if a.shard:
        k, n = (int(v) for v in a.shard.split("/"))
        tids = tids[k::n]

    rows = []
    for tid in tids:
        tg = ts[tid]
        rec_el, rec_xyz = bpose.read_protein_heavy(tg.receptor_pdb)
        rr = np.array([RADII.get(e, 1.9) for e in rec_el])
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
                el = [e for e in g.elements if e != "H"]
                xyz = np.asarray(
                    [c for c, e in zip(g.coords, g.elements, strict=False) if e != "H"],
                    dtype=float)
                if len(el) < 3 or len(el) != len(xyz):
                    continue
                lr = np.array([RADII.get(e, 1.9) for e in el])
                d = np.linalg.norm(xyz[:, None, :] - rec_xyz[None, :, :], axis=2)
                gap = (d - rr[None, :]).min(1) - lr        # surface gap per atom
                rows.append({
                    "tid": tid, "src": nm, "n": len(el),
                    # the shape of the gap distribution, not just its mean
                    "clash": float((gap < -0.4).mean()),      # jammed in
                    "tight": float(((gap >= -0.4) & (gap < 0.4)).mean()),  # optimal
                    "loose": float(((gap >= 0.4) & (gap < 1.5)).mean()),   # weak
                    "solvent": float((gap >= 1.5).mean()),    # contributes ~nothing
                    "gap_med": float(np.median(gap)),
                    "gap_p10": float(np.quantile(gap, 0.1)),
                    "gap_p90": float(np.quantile(gap, 0.9)),
                })
        print(f"{tid[:28]:28s} {len(rows)}", flush=True)
    a.out.write_text(json.dumps(rows))
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()

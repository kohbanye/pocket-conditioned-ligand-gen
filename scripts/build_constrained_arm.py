"""Build a *constrained-sampling* generation arm from an oversampled pool.

Our LM systematically under-generates molecule size (17.6 heavy atoms in the
docked largest fragment vs 22.8 for the CrossDocked reference ligands) and emits
disconnected multi-fragment outputs ~19-32% of the time. Both directly depress
``vina_dock``, which is computed on the largest fragment after re-docking
(measured on the existing 100-pocket dumps: connected & >=20 heavy atoms lifts
separate_4096 from -6.80 to -8.52 kcal/mol).

This script turns an oversampled pool of raw generations into a benchmark arm
under an explicit, reportable inference-time protocol:

1. perceive bond ORDERS from the generated 3D coordinates with Open Babel (our
   own SDF writer emits a single-bond-only bond block, so without this step
   every molecule is aromatic-free and QED/SA are floored);
2. reject a sample unless it sanitizes, forms a single connected component, and
   has ``>= max(min_atoms, round(size_frac * n_ref))`` heavy atoms, where
   ``n_ref`` is the target's reference-ligand heavy-atom count;
3. keep the first ``n_keep`` accepted samples, in generation order.

Rejection is *sampling*, not selection: no property of the accepted molecule
other than connectivity and size enters the decision, and the acceptance test
never looks at a score. Per-target acceptance statistics are written alongside so
the effective sampling cost (pool size / accepted) is reportable.

Run with the sbdd-bench interpreter (it owns the Open Babel binding)::

    <sbdd-bench>/.venv/bin/python scripts/build_constrained_arm.py \
        --pool-dir  <sbdd-bench>/outputs/sep4096_pool/own \
        --out-dir   <sbdd-bench>/outputs/sep4096_cs/own \
        --ref-natoms <scratch>/ref_natoms.json \
        --n-keep 100 --min-atoms 18 --size-frac 0.8 \
        --targets ABL2_HUMAN_274_551_0 ...

Crash isolation: Open Babel perception occasionally segfaults the interpreter, so
the caller is expected to fan targets out over separate processes (``xargs -P``)
rather than looping in one process.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SBDD_BENCH = Path("/gs/bs/tga-ohuelab/sakano/git/sbdd-bench")
sys.path.insert(0, str(SBDD_BENCH))

from rdkit import Chem, RDLogger  # noqa: E402

from sbddbench.molio import obabel_mol  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def _sanitized(mol: Chem.Mol | None) -> Chem.Mol | None:
    if mol is None:
        return None
    try:
        cand = Chem.Mol(mol)
        Chem.SanitizeMol(cand)
    except Exception:  # noqa: BLE001
        return None
    return cand


def _heavy_atoms(mol: Chem.Mol) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def perceive(elements: list[str], coords: np.ndarray) -> Chem.Mol | None:
    """Bond-order perception from 3D coordinates (Open Babel), heavy atoms only."""
    mol = _sanitized(obabel_mol(elements, coords))
    if mol is None:
        return None
    try:
        return Chem.RemoveHs(mol)
    except Exception:  # noqa: BLE001
        return mol


def build_target(  # noqa: PLR0913
    pool_dirs: list[Path],
    out_dir: Path,
    target: str,
    *,
    n_ref: int | None,
    n_keep: int,
    min_atoms: int,
    size_frac: float,
    require_connected: bool,
    fill_remainder: bool,
) -> dict:
    """Filter one target's pool into an arm SDF; returns per-target statistics."""
    jsonls = [p / target / "generated.jsonl" for p in pool_dirs]
    jsonls = [p for p in jsonls if p.exists()]
    if not jsonls:
        return {"target_id": target, "error": "no pool jsonl"}

    # The size floor tracks the target's OWN reference ligand, but never drops
    # below ``min_atoms``: on the smallest-reference third of the test set the
    # baselines run ~1.5x the reference size (DiffSBDD averages 20.9 heavy atoms
    # against a 12.2-atom reference there), and Vina is not size-normalised, so a
    # purely reference-tied floor concedes ~0.7 kcal/mol on those pockets.
    floor = max(int(round(size_frac * n_ref)), min_atoms) if n_ref else min_atoms

    n_pool = n_sanitize = n_connected = 0
    kept: list[Chem.Mol] = []
    atoms_kept: list[int] = []
    # Samples that perceive cleanly but miss the size floor. If the pool runs dry
    # before ``n_keep`` acceptances, the arm is topped up from these (largest
    # first) so every target still contributes exactly ``n_keep`` molecules; the
    # stats record how many slots came from the fallback.
    spare: list[tuple[int, Chem.Mol]] = []
    # Several pool directories are concatenated in order: an oversampled pool can
    # be extended by a further generation run (different --seed) without redoing
    # the first one.
    for pool_idx, jsonl in enumerate(jsonls):
        with jsonl.open() as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("tag") == "ref" or rec.get("idx", 0) < 0:
                    continue
                n_pool += 1
                if len(kept) >= n_keep:
                    continue
                mol = perceive(
                    rec["elements"], np.asarray(rec["coords"], dtype=np.float64)
                )
                if mol is None:
                    continue
                n_sanitize += 1
                frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                if require_connected and len(frags) != 1:
                    continue
                n_connected += 1
                best = max(frags, key=_heavy_atoms)
                n_heavy = _heavy_atoms(best)
                best.SetProp("_Name", f"gen_p{pool_idx}_{rec['idx']}")
                if n_heavy < floor:
                    spare.append((n_heavy, best))
                    continue
                kept.append(best)
                atoms_kept.append(n_heavy)

    n_accepted = len(kept)
    if fill_remainder and len(kept) < n_keep:
        for n_heavy, mol in sorted(spare, key=lambda p: -p[0])[: n_keep - len(kept)]:
            kept.append(mol)
            atoms_kept.append(n_heavy)

    tdir = out_dir / target
    tdir.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(tdir / "generated.sdf")) as w:
        for mol in kept:
            w.write(mol)

    return {
        "target_id": target,
        "n_ref": n_ref or 0,
        "floor": floor,
        "n_pool": n_pool,
        "n_sanitize": n_sanitize,
        "n_connected": n_connected,
        "n_accepted": n_accepted,
        "n_kept": len(kept),
        "n_fallback": len(kept) - n_accepted,
        "filled": int(len(kept) >= n_keep),
        "accept_rate": round(n_accepted / n_pool, 4) if n_pool else 0.0,
        "mean_atoms": round(float(np.mean(atoms_kept)), 2) if atoms_kept else 0.0,
        "error": "",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ref-natoms", type=Path, required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--n-keep", type=int, default=100)
    ap.add_argument("--min-atoms", type=int, default=18)
    ap.add_argument("--size-frac", type=float, default=0.8)
    ap.add_argument("--no-connected", action="store_true")
    ap.add_argument("--fill-remainder", action="store_true")
    ap.add_argument("--stats-dir", type=Path, default=None)
    args = ap.parse_args()

    ref = json.loads(args.ref_natoms.read_text())
    stats_dir = args.stats_dir or (args.out_dir.parent / "arm_stats")
    stats_dir.mkdir(parents=True, exist_ok=True)

    for target in args.targets:
        row = build_target(
            args.pool_dir,
            args.out_dir,
            target,
            n_ref=ref.get(target),
            n_keep=args.n_keep,
            min_atoms=args.min_atoms,
            size_frac=args.size_frac,
            require_connected=not args.no_connected,
            fill_remainder=args.fill_remainder,
        )
        with (stats_dir / f"{target}.csv").open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(row.keys()))
            wr.writeheader()
            wr.writerow(row)
        print(f"[arm] {row}", flush=True)


if __name__ == "__main__":
    main()

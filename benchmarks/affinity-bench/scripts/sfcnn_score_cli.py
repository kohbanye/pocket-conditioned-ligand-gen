"""Score the CASF-2016 core set with Sfcnn's released weights.

Runs inside the Sfcnn environment (see scripts/setup_sfcnn_env.sh). Published
numbers are never copied: a row exists because it was measured here, on the same
crystal poses and labels our arms use.

Sfcnn is a like-for-like peer of our tokenizer in the way that matters for this
bench: it consumes the FULL complex geometry (one joint 20x20x20 A grid centred
on the ligand, holding protein and ligand atoms together) and its input carries
NO explicit interaction features -- the paper states it "simplified the
featurization and only used one-hot encoding of atomic types" (28 channels, 14
protein + 14 ligand). It is the first baseline here with
``uses_explicit_contacts = False`` that also sees the pose.

Its predictions are already on the pK scale (``predict.py`` returns
``model.predict(grid) * 15``), so RMSE is directly comparable to published
figures without recalibration.

Two hazards in upstream's ``predict.py`` are handled here:

* ``build_model()`` calls ``load_weights('weights_22_112-0.0083.h5')`` with a
  bare relative path, so the process must run with the repo root as cwd.
* ``main()`` writes one bare number per ligand with no identifier, so a failure
  would silently shift every later row. This driver keeps pdbid with each
  prediction and reports every failure instead of dropping it.

The paper evaluates on 283 complexes; we score all 285, so our reproduced R will
not equal the published 0.7928. That is the bench contract (same poses, same
labels for every row), not a defect.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def core_set(casf: Path) -> list[dict]:
    """pdbid, logka and cluster for every core complex, from CoreSet.dat."""
    rows = []
    for line in (casf / "power_scoring" / "CoreSet.dat").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        c = line.split()
        if len(c) >= 6:  # noqa: PLR2004
            rows.append({"pdbid": c[0].lower(), "logka": float(c[3]), "cluster": c[5]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--casf", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True, help="Sfcnn checkout root")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ligand-format", default="mol2", help="upstream's own default")
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on N complexes")
    args = ap.parse_args()

    # build_model() resolves the .h5 relative to cwd, so import and call it there.
    sys.path.insert(0, str(args.repo))
    from openbabel import pybel  # noqa: PLC0415
    from predict import build_model, get_grid  # noqa: PLC0415

    model = build_model()

    rows = core_set(args.casf)
    if args.limit:
        rows = rows[: args.limit]
    print(f"scoring {len(rows)} complexes", flush=True)

    suffix = f"_ligand.{args.ligand_format}"
    scored, failed = [], []
    for i, row in enumerate(rows, 1):
        cid = row["pdbid"]
        protein_path = args.casf / "coreset" / cid / f"{cid}_protein.pdb"
        ligand_path = args.casf / "coreset" / cid / f"{cid}{suffix}"
        if not (protein_path.exists() and ligand_path.exists()):
            failed.append((cid, "input file missing"))
            continue
        try:
            protein = next(iter(pybel.readfile("pdb", str(protein_path))))
            ligand = next(iter(pybel.readfile(args.ligand_format, str(ligand_path))))
            grid = get_grid(protein, ligand)
            head = float(model.predict(grid, verbose=0).ravel()[0] * 15)
            scored.append({**row, "head": head})
        except Exception as exc:  # noqa: BLE001 - every failure is reported, never dropped
            failed.append((cid, f"{type(exc).__name__}: {exc}"))
        if i % 25 == 0:
            print(f"  {i}/{len(rows)} ({len(failed)} failed)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdbid", "logka", "cluster", "head"])
        for r in scored:
            w.writerow([r["pdbid"], r["logka"], r["cluster"], r["head"]])

    print(f"\nscored {len(scored)}/{len(rows)} -> {args.out}", flush=True)
    if failed:
        print(f"FAILED {len(failed)}:", flush=True)
        for cid, why in failed[:15]:
            print(f"   {cid}: {why}", flush=True)
    if scored:
        import numpy as np  # noqa: PLC0415

        p = np.array([r["head"] for r in scored])
        y = np.array([r["logka"] for r in scored])
        print(
            f"Pearson R = {np.corrcoef(p, y)[0, 1]:.4f}  "
            f"RMSE = {np.sqrt(((p - y) ** 2).mean()):.3f}  "
            f"pred range {p.min():.2f}..{p.max():.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

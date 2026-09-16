"""Score the CASF-2016 core set with PLANET's released checkpoint.

Runs inside the PLANET environment (third_party/PLANET/.venv-planet). Published
numbers are never copied: a row exists because it was measured here, on the same
crystal poses and labels our arms use.

PLANET's own ``PLANET_run.py`` is deliberately NOT used. Its scoring loop wraps
every batch in ``except: continue``, so a complex it cannot process disappears
from the output instead of being reported. That is not hypothetical here: with
upstream's own ``SDMolSupplier(..., sanitize=True)`` settings, **92 of the 285
CASF ligands fail RDKit sanitisation**, and upstream would have silently scored
the remaining 193 and called the result CASF-2016. This driver reports every
failure and retries the parse instead.

NOTE ON INPUTS: PLANET reads the pocket from the protein PDB plus the crystal
ligand (which only fixes the pocket centre) and represents the ligand as a 2D
graph, so it does NOT consume the ligand's crystal pose -- the authors state the
model "does not take three-dimensional protein-ligand structures as input". The
row belongs beside Boltz-2 as a pose-blind reference, not beside the pose-aware
scoring functions.
"""

from __future__ import annotations

import argparse
import csv
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


def load_ligand(casf: Path, cid: str, source: str):  # noqa: ANN201 - rdkit Mol
    """Parse a CASF ligand with FULL sanitisation, from the requested source.

    92 of the 285 core ligands fail strict sanitisation from the SDF. All 285
    parse strictly from the MOL2, which carries explicit bond orders, so mol2 is
    the default: one uniform route for every complex beats a mixture, and PLANET
    represents the ligand as a 2D GRAPH, so skipping kekulisation or property
    assignment silently degrades exactly the input it depends on. Measured: with
    a partial-sanitisation fallback PLANET scored 0.6562 on those 92 against
    0.7417 on the other 193, while GIGN, OnionNet-2, Sfcnn and Vina -- which read
    coordinates, not graphs -- differed by at most 0.014 between the two groups.
    """
    from rdkit import Chem  # noqa: PLC0415

    if source == "mol2":
        path = casf / "coreset" / cid / f"{cid}_ligand.mol2"
        if not path.exists():
            return None, "mol2 missing"
        mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=True)
        return (mol, "mol2") if mol is not None else (None, "mol2 unparseable")
    path = casf / "coreset" / cid / f"{cid}_ligand.sdf"
    if not path.exists():
        return None, "sdf missing"
    mol = next(iter(Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)), None)
    return (mol, "sdf") if mol is not None else (None, "sdf unparseable")


def score_one(est, feat_fn, casf: Path, cid: str, source: str):  # noqa: ANN001, ANN201
    """One complex -> (affinity, note). Raises nothing; the caller records notes."""
    import torch  # noqa: PLC0415
    from rdkit import Chem  # noqa: PLC0415

    protein = casf / "coreset" / cid / f"{cid}_protein.pdb"
    ligand = casf / "coreset" / cid / f"{cid}_ligand.sdf"
    if not (protein.exists() and ligand.exists()):
        return None, "input file missing"
    mol, note = load_ligand(casf, cid, source)
    if mol is None:
        return None, note
    est.set_pocket_from_ligand(str(protein), str(ligand))
    feat = feat_fn([Chem.AddHs(mol)])
    with torch.no_grad():
        fres, scope = est.model.cal_res_features(est.res_features, 1)
        aff = float(est.model.screening(fres, scope, feat).view([-1]).cpu().numpy()[0])
    return aff, note


def write_dump(out: Path, scored: list[dict]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdbid", "logka", "cluster", "head"])
        for r in scored:
            w.writerow([r["pdbid"], r["logka"], r["cluster"], r["head"]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--casf", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True, help="PLANET checkout root")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on the first N")
    ap.add_argument("--only", default="", help="comma-separated pdbids to score")
    ap.add_argument("--ligand-source", choices=("mol2", "sdf"), default="mol2")
    args = ap.parse_args()

    import sys  # noqa: PLC0415

    sys.path.insert(0, str(args.repo))  # load_parameters() resolves PLANET.param here
    import torch  # noqa: PLC0415
    from chemutils import mol_batch_to_graph  # noqa: PLC0415
    from PLANET_run import PlanetEstimator  # noqa: PLC0415
    from rdkit import RDLogger  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    est = PlanetEstimator(device)
    est.model.to(device)

    rows = core_set(args.casf)
    if args.only:
        want = {x.strip().lower() for x in args.only.split(",") if x.strip()}
        rows = [r for r in rows if r["pdbid"] in want]
    elif args.limit:
        rows = rows[: args.limit]
    print(f"scoring {len(rows)} complexes on {device}", flush=True)

    scored, failed, notes = [], [], {}
    for i, row in enumerate(rows, 1):
        try:
            aff, note = score_one(est, mol_batch_to_graph, args.casf, row["pdbid"], args.ligand_source)
        except Exception as exc:  # noqa: BLE001 - reported, never dropped
            aff, note = None, f"{type(exc).__name__}: {exc}"
        if aff is None:
            failed.append((row["pdbid"], note))
        else:
            scored.append({**row, "head": aff})
            notes[note] = notes.get(note, 0) + 1
        if i % 25 == 0:
            print(f"  {i}/{len(rows)} ({len(failed)} failed)", flush=True)

    write_dump(args.out, scored)
    print(f"\nscored {len(scored)}/{len(rows)} -> {args.out}", flush=True)
    print(f"parse routes: {notes}", flush=True)
    if failed:
        print(f"FAILED {len(failed)}:", flush=True)
        for cid, why in failed[:20]:
            print(f"   {cid}: {why}", flush=True)
    if scored:
        import numpy as np  # noqa: PLC0415

        p = np.array([r["head"] for r in scored])
        y = np.array([r["logka"] for r in scored])
        print(f"Pearson R = {np.corrcoef(p, y)[0, 1]:.4f}  "
              f"RMSE = {np.sqrt(((p - y) ** 2).mean()):.3f}  "
              f"pred range {p.min():.2f}..{p.max():.2f}", flush=True)


if __name__ == "__main__":
    main()

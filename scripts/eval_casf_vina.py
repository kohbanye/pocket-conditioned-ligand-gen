"""AutoDock Vina docking-power baseline on CASF-2016.

For each of the 285 core targets, score every ``decoys_docking`` pose (+ the
crystal native) with ``vina --score_only`` and rank by affinity (most negative =
best). Docking power = fraction of targets whose top-scored pose is within 2 A
RMSD of native (RMSDs from the provided ``_rmsd.dat``). Same pose sets as our
rescorer, for an apples-to-apples comparison.

Receptors are converted to pdbqt once with OpenBabel; poses are scored in
parallel. Run on a many-core CPU node (~28k poses)::

    uv run python scripts/eval_casf_vina.py --workers 40 \
        --out-csv outputs/casf/vina.csv
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

from scripts.dock_vina import _parse_score, _run, _write_xyz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_VINA = "/home/5/uq02055/.local/bin/vina"
DEFAULT_OBABEL = "/home/5/uq02055/usr/app/babel/bin/obabel"
_CFG: dict = {}


def _init(cfg: dict) -> None:
    global _CFG  # noqa: PLW0603
    _CFG = cfg


def _mol_atoms(mol) -> tuple[list[str], list[list[float]]]:  # noqa: ANN001
    conf = mol.GetConformer()
    els, xyz = [], []
    for i, a in enumerate(mol.GetAtoms()):
        p = conf.GetAtomPosition(i)
        els.append(a.GetSymbol())
        xyz.append([p.x, p.y, p.z])
    return els, xyz


def _score_pose(rec: dict) -> dict:
    cfg = _CFG
    els, coords = rec["elements"], rec["coords"]
    n = len(els)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]
    cx, cy, cz = sum(xs) / n, sum(ys) / n, sum(zs) / n
    margin, floor = 8.0, 22.5
    box = [
        "--center_x",
        f"{cx:.3f}",
        "--center_y",
        f"{cy:.3f}",
        "--center_z",
        f"{cz:.3f}",
        "--size_x",
        f"{max(floor, max(xs) - min(xs) + margin):.3f}",
        "--size_y",
        f"{max(floor, max(ys) - min(ys) + margin):.3f}",
        "--size_z",
        f"{max(floor, max(zs) - min(zs) + margin):.3f}",
    ]
    out = {"tid": rec["tid"], "name": rec["name"], "rmsd": rec["rmsd"], "score": None}
    with tempfile.TemporaryDirectory(dir=cfg["tmp_dir"]) as td:
        xyz = Path(td) / "lig.xyz"
        pdbqt = Path(td) / "lig.pdbqt"
        _write_xyz(xyz, els, coords)
        _run(
            [
                cfg["obabel"],
                str(xyz),
                "-O",
                str(pdbqt),
                "-r",
                "-p",
                "7.4",
                "--partialcharge",
                "gasteiger",
            ]
        )
        if not pdbqt.exists() or "ATOM" not in pdbqt.read_text()[:99999]:
            return out
        try:
            r = _run(
                [
                    cfg["vina"],
                    "--receptor",
                    rec["receptor"],
                    "--ligand",
                    str(pdbqt),
                    "--score_only",
                    "--cpu",
                    "1",
                    "--seed",
                    "1",
                    *box,
                ]
            )
            out["score"] = _parse_score(r.stdout)
        except Exception:  # noqa: BLE001
            return out
    return out


def _prep_receptor(protein_pdb: Path, out_pdbqt: Path, obabel: str) -> bool:
    if out_pdbqt.exists() and out_pdbqt.stat().st_size > 0:
        return True
    # Plain rigid-receptor pdbqt. Do NOT add "-p 7.4 --partialcharge gasteiger":
    # protonating a whole protein makes obabel emit "0 molecules converted" for
    # ~29% of CASF proteins. Vina assigns its own receptor terms anyway.
    _run([obabel, str(protein_pdb), "-O", str(out_pdbqt), "-xr"], timeout=300)
    return out_pdbqt.exists() and out_pdbqt.stat().st_size > 0


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--casf-dir", type=Path, default=Path("data/casf2016"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/casf/vina.csv"))
    parser.add_argument(
        "--rec-dir", type=Path, default=Path("outputs/casf/vina_receptors")
    )
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--vina", type=str, default=DEFAULT_VINA)
    parser.add_argument("--obabel", type=str, default=DEFAULT_OBABEL)
    parser.add_argument("--native-thresh", type=float, default=2.0)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument(
        "--exclude-native",
        action="store_true",
        help="Rank docking decoys only (CASF-standard; matches our --exclude-native).",
    )
    parser.add_argument("--tmp-dir", type=str, default=str(Path.home() / "tmpdir"))
    parser.add_argument(
        "--dump-scores", type=Path, default=None, help="Per-pose Vina scores CSV."
    )
    args = parser.parse_args()

    from rdkit import Chem, RDLogger  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    args.rec_dir.mkdir(parents=True, exist_ok=True)

    from scripts.eval_casf_rescore import _parse_mol2_multi  # noqa: PLC0415

    targets = sorted(
        p.name for p in (args.casf_dir / "coreset").iterdir() if p.is_dir()
    )
    if args.max_targets is not None:
        targets = targets[: args.max_targets]

    records: list[dict] = []
    for tid in targets:
        protein = args.casf_dir / "coreset" / tid / f"{tid}_protein.pdb"
        decoys = args.casf_dir / "decoys_docking" / f"{tid}_decoys.mol2"
        rmsd_dat = args.casf_dir / "decoys_docking" / f"{tid}_rmsd.dat"
        native_mol2 = args.casf_dir / "coreset" / tid / f"{tid}_ligand.mol2"
        rec_pdbqt = args.rec_dir / f"{tid}.pdbqt"
        if not (protein.exists() and decoys.exists() and rmsd_dat.exists()):
            continue
        if not _prep_receptor(protein, rec_pdbqt, args.obabel):
            logger.warning("receptor prep failed: %s", tid)
            continue
        rmsd = {}
        for ln in rmsd_dat.read_text().splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            nm, val = ln.split()[:2]
            rmsd[nm] = float(val)
        # crystal native (RMSD 0), unless decoys-only
        nat = (
            None
            if args.exclude_native
            else Chem.MolFromMol2File(str(native_mol2), sanitize=False, removeHs=False)
        )
        if nat is not None:
            els, xyz = _mol_atoms(nat)
            records.append(
                {
                    "tid": tid,
                    "name": f"{tid}_native",
                    "rmsd": 0.0,
                    "elements": els,
                    "coords": xyz,
                    "receptor": str(rec_pdbqt),
                }
            )
        # decoys: _parse_mol2_multi returns each pose's atoms directly.
        for name, d in _parse_mol2_multi(decoys.read_text()):
            if name not in rmsd:
                continue
            els = [a[0] for a in d["atoms"]]
            xyz = [[a[1], a[2], a[3]] for a in d["atoms"]]
            records.append(
                {
                    "tid": tid,
                    "name": name,
                    "rmsd": rmsd[name],
                    "elements": els,
                    "coords": xyz,
                    "receptor": str(rec_pdbqt),
                }
            )
    logger.info("scoring %d poses over %d targets", len(records), len(targets))

    import multiprocessing  # noqa: PLC0415

    from tqdm import tqdm  # noqa: PLC0415

    cfg = {"vina": args.vina, "obabel": args.obabel, "tmp_dir": args.tmp_dir}
    with multiprocessing.Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_score_pose, records, chunksize=8),
                total=len(records),
                desc="poses",
            )
        )

    by_target: dict[str, list[dict]] = {}
    for r in results:
        by_target.setdefault(r["tid"], []).append(r)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.dump_scores is not None:
        with args.dump_scores.open("w") as f:
            f.write("pdbid,pose,rmsd,head\n")
            for r in results:
                if r["score"] is not None:
                    # head = -affinity so higher = better (matches ensemble tools)
                    f.write(f"{r['tid']},{r['name']},{r['rmsd']:.3f},{-r['score']:.4f}\n")
        logger.info("wrote per-pose Vina scores to %s", args.dump_scores)
    successes = scored = 0
    with args.out_csv.open("w") as f:
        f.write("pdbid,top_pose_rmsd,success,n_poses\n")
        for tid, rs in sorted(by_target.items()):
            scoredr = [r for r in rs if r["score"] is not None]
            if len(scoredr) < 3:  # noqa: PLR2004
                continue
            top = min(scoredr, key=lambda r: r["score"])  # most negative affinity
            ok = int(top["rmsd"] <= args.native_thresh)
            successes += ok
            scored += 1
            f.write(f"{tid},{top['rmsd']:.3f},{ok},{len(scoredr)}\n")
    logger.info(
        "=== Vina CASF docking power (top1<=%.1fA): %d/%d = %.1f%% ===",
        args.native_thresh,
        successes,
        scored,
        100 * successes / max(1, scored),
    )


if __name__ == "__main__":
    main()

"""Prepare docking targets for the benchmark.

For each target this writes, under ``data/targets/<tag>/``:

    <tag>_receptor.pdb      protein ATOM records (standard amino acids only)
    <tag>_pocket.pdb        residues with any atom ≤ --pocket-cutoff Å of the
                            reference ligand (TargetDiff / DiffGui condition on
                            this directly)
    <tag>_ref_ligand.sdf    reference ligand (bonds perceived by Open Babel)
    <tag>_receptor.pdbqt    receptor prepared for AutoDock Vina
    <tag>_box.json          {"center": [...], "size": [...]} docking box

and appends a record to ``data/targets/index.json``.

Input modes
-----------
* single crystal structure::

    python scripts/prepare_targets.py --pdb-id 2ITY --ligand-resname IRE --tag 2ity

* a batch JSON (one object per target), each either a PDB to split
  ``{"tag","pdb_id"|"pdb_file","ligand_resname"?,"chain"?}`` or an already-split
  pair ``{"tag","receptor_pdb","ref_ligand_sdf"}``::

    python scripts/prepare_targets.py --pairs my_targets.json

* a CrossDocked-style test folder of ``*_pocket*.pdb`` + matching ``*.sdf``
  (the 100-pocket SBDD test set)::

    python scripts/prepare_targets.py --crossdocked-test data/crossdocked_test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sbddbench import paths  # noqa: E402

_STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}
_WATER = {"HOH", "WAT", "DOD"}


def _run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, check=False)


def download_pdb(pdb_id: str, dest: Path) -> None:
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
        dest.write_bytes(resp.read())


def _atom_xyz(ln: str) -> tuple[float, float, float]:
    return float(ln[30:38]), float(ln[38:46]), float(ln[46:54])


def split_from_pdb(raw_pdb: Path, out_dir: Path, tag: str, resname: str | None,
                   chain: str | None) -> tuple[Path, Path]:
    """Split a crystal PDB into receptor.pdb + ref_ligand_raw.pdb."""
    lines = raw_pdb.read_text().splitlines()
    atoms = [ln for ln in lines if ln[:6].strip() == "ATOM"]
    hets = [ln for ln in lines if ln[:6].strip() == "HETATM"]
    if chain:
        atoms = [ln for ln in atoms if ln[21] == chain]
        hets = [ln for ln in hets if ln[21] == chain]
    if resname is None:
        counts = Counter(ln[17:20].strip() for ln in hets)
        for w in _WATER:
            counts.pop(w, None)
        if not counts:
            raise SystemExit(f"{tag}: no non-water HETATM to use as reference ligand")
        resname = counts.most_common(1)[0][0]
    lig = [ln for ln in hets if ln[17:20].strip() == resname]
    if not lig:
        raise SystemExit(f"{tag}: ligand {resname!r} not found")
    c0, r0 = lig[0][21], lig[0][22:26]
    lig = [ln for ln in lig if ln[21] == c0 and ln[22:26] == r0]
    rec = [ln for ln in atoms if ln[17:20].strip() in _STANDARD_AA]

    receptor_pdb = out_dir / f"{tag}_receptor.pdb"
    ligand_pdb = out_dir / f"{tag}_ref_ligand_raw.pdb"
    receptor_pdb.write_text("\n".join(rec) + "\nTER\nEND\n")
    ligand_pdb.write_text("\n".join(lig) + "\nEND\n")
    return receptor_pdb, ligand_pdb


def ligand_to_sdf(ligand_in: Path, out_sdf: Path) -> None:
    r = _run([paths.OBABEL, ligand_in, "-O", out_sdf])
    if not out_sdf.exists():
        print("  obabel sdf stderr:", r.stderr.strip()[:200])


def ligand_coords(ref_sdf: Path) -> np.ndarray:
    from rdkit import Chem

    mol = next(iter(Chem.SDMolSupplier(str(ref_sdf), sanitize=False, removeHs=True)), None)
    if mol is None or mol.GetNumConformers() == 0:
        raise SystemExit(f"could not read reference ligand coords from {ref_sdf}")
    conf = mol.GetConformer()
    return np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                     for i in range(mol.GetNumAtoms())])


def write_box(out_json: Path, lig_xyz: np.ndarray, padding: float, min_edge: float,
              resname: str | None) -> dict:
    center = lig_xyz.mean(0)
    extent = lig_xyz.max(0) - lig_xyz.min(0)
    size = np.maximum(extent + 2 * padding, min_edge)
    box = {"center": [round(float(c), 3) for c in center],
           "size": [round(float(s), 3) for s in size]}
    if resname:
        box["ligand_resname"] = resname
    out_json.write_text(json.dumps(box, indent=2))
    return box


def write_pocket_pdb(receptor_pdb: Path, lig_xyz: np.ndarray, out_pocket: Path,
                     cutoff: float) -> int:
    """Keep residues with any atom within ``cutoff`` Å of the ligand."""
    lines = receptor_pdb.read_text().splitlines()
    residues: dict = {}
    for ln in lines:
        if ln[:6].strip() not in ("ATOM", "HETATM"):
            continue
        key = (ln[21], ln[22:27])
        residues.setdefault(key, []).append(ln)
    keep = []
    cutoff2 = cutoff * cutoff
    for _key, res_lines in residues.items():
        coords = np.array([_atom_xyz(ln) for ln in res_lines])
        d2 = ((coords[:, None, :] - lig_xyz[None, :, :]) ** 2).sum(-1).min()
        if d2 <= cutoff2:
            keep.extend(res_lines)
    out_pocket.write_text("\n".join(keep) + "\nTER\nEND\n")
    return len({(ln[21], ln[22:27]) for ln in keep})


def receptor_to_pdbqt(receptor_pdb: Path, out_pdbqt: Path) -> bool:
    # Success is judged by the output file, not the exit status: prepare_receptor
    # returns 0 on inputs it silently declines to convert.
    _run([paths.PREPARE_RECEPTOR, "-r", receptor_pdb, "-o", out_pdbqt,
          "-A", "checkhydrogens", "-U", "nphs_lps_waters_nonstdres"])
    if out_pdbqt.exists():
        return True
    print("  prepare_receptor failed; obabel fallback")
    _run([paths.OBABEL, receptor_pdb, "-O", out_pdbqt, "-xr", "-p", "7.4",
          "--partialcharge", "gasteiger"])
    return out_pdbqt.exists()


def prepare_one(record: dict, args) -> dict | None:
    tag = record["tag"]
    out_dir = paths.TARGETS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{tag}]")

    # --- obtain receptor.pdb + ref_ligand.sdf ---
    if record.get("receptor_pdb") and record.get("ref_ligand_sdf"):
        receptor_pdb = Path(record["receptor_pdb"])
        ref_sdf = out_dir / f"{tag}_ref_ligand.sdf"
        src = Path(record["ref_ligand_sdf"])
        if src.suffix.lower() != ".sdf":
            ligand_to_sdf(src, ref_sdf)
        else:
            ref_sdf.write_bytes(src.read_bytes())
        # copy receptor into the target dir for a self-contained index
        local_rec = out_dir / f"{tag}_receptor.pdb"
        if receptor_pdb.resolve() != local_rec.resolve():
            local_rec.write_bytes(receptor_pdb.read_bytes())
        receptor_pdb = local_rec
        resname = record.get("ligand_resname")
    else:
        raw = out_dir / f"{tag}_raw.pdb"
        if record.get("pdb_file"):
            raw.write_bytes(Path(record["pdb_file"]).read_bytes())
        else:
            download_pdb(record["pdb_id"], raw)
        receptor_pdb, ligand_pdb = split_from_pdb(
            raw, out_dir, tag, record.get("ligand_resname"), record.get("chain"))
        ref_sdf = out_dir / f"{tag}_ref_ligand.sdf"
        ligand_to_sdf(ligand_pdb, ref_sdf)
        resname = record.get("ligand_resname")

    if not ref_sdf.exists():
        print(f"  SKIP {tag}: no reference ligand SDF")
        return None

    lig_xyz = ligand_coords(ref_sdf)
    box = write_box(out_dir / f"{tag}_box.json", lig_xyz, args.box_padding, args.min_box, resname)
    n_pocket = write_pocket_pdb(receptor_pdb, lig_xyz, out_dir / f"{tag}_pocket.pdb", args.pocket_cutoff)
    receptor_pdbqt = out_dir / f"{tag}_receptor.pdbqt"
    ok_pdbqt = receptor_to_pdbqt(receptor_pdb, receptor_pdbqt)
    print(f"  pocket residues: {n_pocket} | box {box['center']} {box['size']} | pdbqt {'ok' if ok_pdbqt else 'FAILED'}")

    return {
        "target_id": tag,
        "receptor_pdb": str(receptor_pdb.relative_to(paths.TARGETS_DIR)),
        "pocket_pdb": str((out_dir / f"{tag}_pocket.pdb").relative_to(paths.TARGETS_DIR)),
        "ref_ligand_sdf": str(ref_sdf.relative_to(paths.TARGETS_DIR)),
        "receptor_pdbqt": str(receptor_pdbqt.relative_to(paths.TARGETS_DIR)) if ok_pdbqt else None,
        "box": box,
        "meta": record.get("meta", {}),
    }


def collect_records(args) -> list[dict]:
    if args.pairs:
        return json.loads(Path(args.pairs).read_text())
    if args.crossdocked_test:
        root = Path(args.crossdocked_test)
        recs = []
        pockets = sorted(root.rglob("*_pocket*.pdb"))
        if pockets:
            for pocket in pockets:
                stem = pocket.name.split("_pocket")[0]
                sdf = next(iter(pocket.parent.glob(f"{stem}*.sdf")), None)
                if sdf is None:
                    continue
                recs.append({"tag": f"{pocket.parent.name}_{stem}".replace("/", "-"),
                             "receptor_pdb": str(pocket), "ref_ligand_sdf": str(sdf)})
        else:
            # TargetDiff/DiffSBDD CrossDocked2020 test_set layout: per-target dirs
            # holding a full receptor "<stem>_rec.pdb" + its reference ligand
            # "<stem>_rec_*.sdf" (prepare_one carves the pocket around the ref
            # ligand). 7 dirs hold TWO (receptor, ligand) pairs -> 100 pockets from
            # 93 dirs. Pair each receptor with ITS OWN ligand by shared stem, and
            # give multi-pair dirs unique tags ("<dir>__<stem>"); single-pair dirs
            # keep the plain dir-name tag.
            for rec_pdb in sorted(root.rglob("*_rec.pdb")):
                stem = rec_pdb.name[: -len(".pdb")]
                sdf = next(iter(sorted(rec_pdb.parent.glob(f"{stem}*.sdf"))), None)
                if sdf is None:
                    sdf = next(iter(sorted(rec_pdb.parent.glob("*.sdf"))), None)
                if sdf is None:
                    continue
                multi = len(list(rec_pdb.parent.glob("*_rec.pdb"))) > 1
                tag = f"{rec_pdb.parent.name}__{stem}" if multi else rec_pdb.parent.name
                recs.append({"tag": tag.replace("/", "-"),
                             "receptor_pdb": str(rec_pdb), "ref_ligand_sdf": str(sdf)})
        return recs
    if args.pdb_id or args.pdb_file:
        return [{"tag": args.tag, "pdb_id": args.pdb_id, "pdb_file": args.pdb_file,
                 "ligand_resname": args.ligand_resname, "chain": args.chain}]
    raise SystemExit("provide --pdb-id/--pdb-file, --pairs, or --crossdocked-test")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdb-id")
    p.add_argument("--pdb-file")
    p.add_argument("--ligand-resname", default=None)
    p.add_argument("--chain", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--pairs", help="batch JSON of target records")
    p.add_argument("--crossdocked-test", help="folder of *_pocket*.pdb + *.sdf pairs")
    p.add_argument("--pocket-cutoff", type=float, default=10.0)
    p.add_argument("--box-padding", type=float, default=6.0)
    p.add_argument("--min-box", type=float, default=22.5)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    if (args.pdb_id or args.pdb_file) and not args.tag:
        args.tag = (args.pdb_id or Path(args.pdb_file).stem).lower()

    paths.ensure_dirs()
    records = collect_records(args)
    if args.limit:
        records = records[: args.limit]
    print(f"preparing {len(records)} target(s)")

    index_path = paths.TARGETS_DIR / "index.json"
    existing = json.loads(index_path.read_text()) if index_path.exists() else []
    by_id = {r["target_id"]: r for r in existing}
    for rec in records:
        try:
            out = prepare_one(rec, args)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {rec.get('tag')}: {exc!r}")
            continue
        if out:
            by_id[out["target_id"]] = out
    index = list(by_id.values())
    index_path.write_text(json.dumps(index, indent=2))
    print(f"\nwrote {len(index)} targets -> {index_path}")


if __name__ == "__main__":
    main()

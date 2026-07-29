"""Prepare CrossDocked *train* pockets as generation targets for refiner training.

The distillation set must not come from the pockets the refiner is later scored
on. This pulls pockets from the CrossDocked train split in the source repo's
``hub_cache`` and writes the two files the generation + relaxation pipeline needs
per target — a receptor PDB and a reference-ligand SDF — plus a target index in
the sbdd-bench schema. No docking box or receptor PDBQT is produced: training
poses are never scored, so the Vina-side preparation is unnecessary.

Target ids that appear in the evaluation index are skipped explicitly, so the two
sets are disjoint by construction rather than by assumption.

    python scripts/prepare_train_pockets.py --n 80 --out-dir <sbdd-bench>/data/train_pockets
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

SOURCE_REPO = Path("/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen")
SBDD_BENCH = Path("/gs/bs/tga-ohuelab/sakano/git/sbdd-bench")
sys.path.insert(0, str(SOURCE_REPO))

from rdkit import Chem, RDLogger  # noqa: E402
from scripts.generate_ligands_3d import _read_mol_from_tar  # noqa: E402

RDLogger.DisableLog("rdApp.*")

_BOND_ORDER = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}


def mol_to_sdf(mol: dict, path: Path) -> bool:
    """Write the stored ligand (atoms + bond orders) as an SDF."""
    rw = Chem.RWMol()
    conf_pos = []
    idx_map = {}
    for i, (el, x, y, z) in enumerate(mol["atoms"]):
        if el == "H":
            continue
        idx_map[i] = rw.AddAtom(Chem.Atom(el))
        conf_pos.append((float(x), float(y), float(z)))
    if len(conf_pos) < 3:  # noqa: PLR2004
        return False
    for a, b, t in mol["bonds"]:
        if a in idx_map and b in idx_map and a != b:
            try:
                rw.AddBond(idx_map[a], idx_map[b], _BOND_ORDER.get(t, Chem.BondType.SINGLE))
            except Exception:  # noqa: BLE001
                pass
    m = rw.GetMol()
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, p in enumerate(conf_pos):
        conf.SetAtomPosition(i, p)
    m.AddConformer(conf)
    try:
        Chem.SanitizeMol(m)
    except Exception:  # noqa: BLE001
        return False
    with Chem.SDWriter(str(path)) as w:
        w.write(m)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--out-dir", type=Path, default=SBDD_BENCH / "data" / "train_pockets")
    ap.add_argument(
        "--exclude-index",
        type=Path,
        default=SBDD_BENCH / "data" / "targets" / "index.json",
    )
    args = ap.parse_args()

    excl = json.loads(args.exclude_index.read_text())
    excl_ids = {
        t["target_id"] for t in (excl["targets"] if isinstance(excl, dict) else excl)
    }

    hub = SOURCE_REPO / "data" / "hub_cache"
    mdf = pq.read_table(hub / "repo" / "manifest.parquet").to_pandas()
    tr = mdf[(mdf.source_type == "cdonly") & (mdf.cdonly_fold0 == "train")]
    # One row per pocket: the first pose is enough, the ligand only defines where
    # the pocket is.
    first = tr.groupby("complex_dir", sort=True).first().reset_index()
    first = first[~first.complex_dir.isin(excl_ids)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for _, row in first.iterrows():
        if len(records) >= args.n:
            break
        tid = str(row["complex_dir"])
        rec_src = hub / "receptors" / tid / str(row["receptor_pdb"])
        if not rec_src.exists():
            continue
        mol = _read_mol_from_tar(hub / "repo", int(row["shard_idx"]), int(row["pair_idx"]))
        if mol is None:
            continue
        tdir = args.out_dir / tid
        tdir.mkdir(parents=True, exist_ok=True)
        rec_dst = tdir / f"{tid}_receptor.pdb"
        lig_dst = tdir / f"{tid}_ref_ligand.sdf"
        if not mol_to_sdf(mol, lig_dst):
            continue
        if not rec_dst.exists():
            shutil.copyfile(rec_src, rec_dst)
        records.append(
            {
                "target_id": tid,
                "receptor_pdb": f"{tid}/{tid}_receptor.pdb",
                "pocket_pdb": f"{tid}/{tid}_receptor.pdb",
                "ref_ligand_sdf": f"{tid}/{tid}_ref_ligand.sdf",
                "receptor_pdbqt": "",
                "box": {"center": [0.0, 0.0, 0.0], "size": [22.5, 22.5, 22.5]},
                "meta": {"split": "cdonly_fold0_train"},
            }
        )
        print(f"[prep] {tid}", flush=True)

    (args.out_dir / "index.json").write_text(json.dumps(records, indent=1))
    ids = args.out_dir / "target_ids.txt"
    ids.write_text("\n".join(r["target_id"] for r in records) + "\n")
    print(f"prepared {len(records)} train pockets -> {args.out_dir}")
    print(f"overlap with eval set: {len({r['target_id'] for r in records} & excl_ids)}")


if __name__ == "__main__":
    main()

"""Score the CASF-2016 core set with GIGN's released checkpoints.

Runs inside the GIGN environment (see scripts/setup_gign_env.sh), driven by
affinity_bench.adapters.gign. Published numbers are never copied: a row exists
because it was measured here, on the same poses and labels our arms use.

Inputs are staged as real directories holding SYMLINKS to the CASF files.
GIGN's preprocessing writes `<cid>_ligand.pdb`, `Pocket_5A.pdb`, the `.rdkit`
pickle and the `.pyg` graph NEXT TO its inputs, and data/casf2016 has to stay
pristine.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


def stage(casf: Path, work: Path) -> tuple[Path, Path]:
    """Symlinked copy of the core set, plus the label CSV GIGN's loader wants."""
    data_dir = work / "data" / "test2016"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in (casf / "power_scoring" / "CoreSet.dat").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        c = line.split()
        if len(c) >= 6:  # noqa: PLR2004
            rows.append({"pdbid": c[0].lower(), "-logKd/Ki": float(c[3]), "cluster": c[5]})
    for row in rows:
        cid = row["pdbid"]
        dst = data_dir / cid
        dst.mkdir(exist_ok=True)
        for suffix in ("_ligand.mol2", "_protein.pdb"):
            link, target = dst / f"{cid}{suffix}", casf / "coreset" / cid / f"{cid}{suffix}"
            if not target.exists():
                msg = f"missing CASF input: {target}"
                raise FileNotFoundError(msg)
            if not link.exists():
                link.symlink_to(target)
    csv_path = work / "data" / "test2016.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["pdbid", "-logKd/Ki", "cluster"])
        writer.writeheader()
        writer.writerows(rows)
    return data_dir, csv_path


def main() -> None:  # noqa: PLR0915 - a driver is a sequence of steps
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--casf", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=None, help="GIGN submodule root")
    # int, not float: the cutoff is formatted into the cached filenames and the
    # dataset re-derives the same name, so 5.0 makes it miss every file the
    # preprocessing just wrote.
    ap.add_argument("--distance", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    repo = args.repo or Path(__file__).resolve().parents[3] / "third_party" / "GIGN"
    src = repo / "GIGN"
    sys.path.insert(0, str(src))
    os.chdir(src)  # their modules use relative imports and paths

    # Imported here, not at module scope: these live in the GIGN submodule
    # and only resolve after the sys.path insert above.
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from dataset_GIGN import GraphDataset, PLIDataLoader  # noqa: PLC0415
    from GIGN import GIGN  # noqa: PLC0415
    from preprocessing import generate_complex, generate_pocket  # noqa: PLC0415

    data_dir, csv_path = stage(args.casf.resolve(), args.work.resolve())
    df = pd.read_csv(csv_path)

    generate_pocket(data_dir=str(data_dir), distance=args.distance)
    generate_complex(str(data_dir), df, distance=args.distance, input_ligand_format="mol2")
    dataset = GraphDataset(str(data_dir), df, graph_type="Graph_GIGN",
                           dis_threshold=args.distance, create=True)
    if len(dataset) != len(df):
        msg = f"{len(df) - len(dataset)} complexes produced no graph; refusing a partial dump"
        raise RuntimeError(msg)
    loader = PLIDataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    ckpts = sorted((src / "model").glob("*/model/*.pt"))
    if not ckpts:
        msg = f"no checkpoints under {src / 'model'}"
        raise FileNotFoundError(msg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} checkpoints={len(ckpts)}", flush=True)

    for ckpt in ckpts:
        repeat = ckpt.parent.parent.name
        model = GIGN(35, 256).to(device)
        model.load_state_dict(torch.load(str(ckpt), map_location=device))
        model.eval()
        preds: list[float] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)  # noqa: PLW2901
                preds.extend(model(batch).detach().cpu().numpy().ravel().tolist())
        if len(preds) != len(df):
            msg = f"{repeat}: {len(preds)} predictions for {len(df)} complexes"
            raise RuntimeError(msg)
        out = args.out_dir / f"{repeat}.csv"
        with out.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["pdbid", "logka", "cluster", "head"])
            # No strict=: GIGN's env is Python 3.9. Lengths are checked above.
            for (_i, row), pred in zip(df.iterrows(), preds):  # noqa: B905
                writer.writerow([row["pdbid"], row["-logKd/Ki"], row["cluster"], pred])
        r = np.corrcoef(np.asarray(preds), df["-logKd/Ki"].to_numpy())[0, 1]
        print(f"{repeat}: n={len(preds)} Pearson R={r:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()

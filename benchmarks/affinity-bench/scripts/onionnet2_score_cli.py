"""Score the CASF-2016 core set with OnionNet-2's released model.

Runs inside the OnionNet-2 environment (see scripts/setup_onionnet2_env.sh),
driven by affinity_bench.adapters.onionnet2.

Upstream's `scoring/predict.py` handles ONE complex per process and reloads the
885 MB model each time. This imports the same feature code and loads the model
and scaler once; features, shape, scaler and custom objects are theirs.

Two names that predict.py builds only under `__main__` are read as globals by
its importable feature code -- `args` (for args.shells) and `keys` (the feature
column order). Importing the module leaves both unset and every complex dies
with NameError, so both are supplied here rather than by editing their file.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
from pathlib import Path

#: Copied verbatim from upstream's main. The product ORDER is load-bearing: it
#: fixes which feature column is which, and the released scaler and model were
#: fitted against that order.
LIG_ELEMENTS = ["H", "C", "O", "N", "P", "S", "Hal", "DU"]
REC_RESIDUES = [
    "GLY", "ALA", "VAL", "LEU", "ILE", "PRO", "PHE", "TYR", "TRP", "SER",
    "THR", "CYS", "MET", "ASN", "GLN", "ASP", "GLU", "LYS", "ARG", "HIS", "OTH",
]


def load_labels(casf: Path) -> dict[str, tuple[float, str]]:
    out: dict[str, tuple[float, str]] = {}
    for line in (casf / "power_scoring" / "CoreSet.dat").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        c = line.split()
        if len(c) >= 6:  # noqa: PLR2004
            out[c[0].lower()] = (float(c[3]), c[5])
    return out


def main() -> None:  # noqa: PLR0915 - a driver is a sequence of steps
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--casf", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--scaler", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=None, help="OnionNet-2 submodule root")
    ap.add_argument("--shells", type=int, default=62)
    ap.add_argument("--shape", type=str, default="84,124,1")
    ap.add_argument("--obabel", type=str, default="obabel")
    args = ap.parse_args()

    repo = args.repo or Path(__file__).resolve().parents[3] / "third_party" / "OnionNet-2"
    sys.path.insert(0, str(repo / "scoring"))

    # Imported here, not at module scope: `predict` lives in the OnionNet-2
    # submodule and only resolves after the sys.path insert above.
    import joblib  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import predict as onion  # noqa: PLC0415
    import tensorflow as tf  # noqa: PLC0415

    onion.args = argparse.Namespace(shells=args.shells)
    onion.keys = ["_".join(x) for x in itertools.product(REC_RESIDUES, LIG_ELEMENTS)]

    shape = [int(x) for x in args.shape.split(",")]
    # 21 residue types x 8 ligand elements = 168 per shell; 168 x 62 = 10,416
    # = 84 x 124, the shape the released model expects.
    if len(onion.keys) * args.shells != shape[0] * shape[1]:
        msg = "feature layout does not match the released model's shape"
        raise ValueError(msg)

    labels = load_labels(args.casf)
    lig_dir = args.work / "ligand_pdb"
    lig_dir.mkdir(parents=True, exist_ok=True)

    scaler = joblib.load(str(args.scaler))
    model = tf.keras.models.load_model(
        str(args.model),
        custom_objects={"RMSE": onion.RMSE, "PCC": onion.PCC, "PCC_RMSE": onion.PCC_RMSE},
    )
    print(f"model and scaler loaded once; {len(labels)} complexes", flush=True)

    rows, failed = [], []
    for n, (pdbid, (logka, cluster)) in enumerate(sorted(labels.items()), 1):
        rec = args.casf / "coreset" / pdbid / f"{pdbid}_protein.pdb"
        mol2 = args.casf / "coreset" / pdbid / f"{pdbid}_ligand.mol2"
        lig = lig_dir / f"{pdbid}_ligand.pdb"
        try:
            if not lig.exists():
                # ParseLigand reads ATOM/HETATM records, so the ligand must be
                # PDB; upstream's README converts with OpenBabel too.
                subprocess.run([args.obabel, str(mol2), "-O", str(lig)],  # noqa: S603
                               check=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            feat = onion.generate_features(str(rec), str(lig))
            x = scaler.transform(np.asarray(feat).reshape(1, -1)).reshape([-1, *shape])
            pred = float(model.predict(x).ravel()[0])
        except Exception as exc:  # noqa: BLE001
            failed.append((pdbid, repr(exc)[:120]))
            print(f"[{n}/{len(labels)}] {pdbid} FAILED: {exc!r:.120}", flush=True)
            continue
        rows.append({"pdbid": pdbid, "logka": logka, "cluster": cluster, "head": pred})
        if n % 25 == 0:
            print(f"[{n}/{len(labels)}] {pdbid} pred={pred:.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["pdbid", "logka", "cluster", "head"])
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        p = np.array([r["head"] for r in rows])
        y = np.array([r["logka"] for r in rows])
        print(f"scored {len(rows)}/{len(labels)} Pearson R={np.corrcoef(p, y)[0, 1]:.4f}"
              f" -> {args.out}", flush=True)
    # Silence is not success: a partial run has to be visible, not averaged over.
    if failed:
        print(f"FAILED {len(failed)}: {failed[:10]}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

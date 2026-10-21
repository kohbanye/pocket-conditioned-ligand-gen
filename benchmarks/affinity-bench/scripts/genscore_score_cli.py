"""Score the CASF-2016 core set with GenScore's released GatedGCN checkpoint.

Runs inside the GenScore environment (a conda env carrying its pinned DGL
build), driven by affinity_bench.adapters.genscore.

The model construction, hyperparameters and pocket extraction are GenScore's
own; this file is the glue that was previously a loose harness in a sibling
checkout, repointed at the pinned submodule. The hyperparameters below are
copied verbatim -- they are not defaults, they define the released model's
architecture, and a mismatch loads weights into the wrong shape.

Two core complexes, 4k18 and 4mme, fail upstream graph construction from the
distributed inputs. They are recovered by repairing only those inputs (see
--repaired-dir); without them this is a 283-complex comparison, and differencing
a 283-complex correlation against a 285-complex one is not a comparison at all.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

#: GenScore's released architecture. Verbatim from the upstream harness.
MP = {
    "batch_size": 128,
    "dist_threhold": 5.0,
    "num_node_featsp": 41,
    "num_node_featsl": 41,
    "num_edge_featsp": 5,
    "num_edge_featsl": 10,
    "hidden_dim0": 128,
    "hidden_dim": 128,
    "n_gaussians": 10,
    "dropout_rate": 0.15,
}


def setup_babel() -> None:
    """Point OpenBabel at its data directory when the env has not."""
    if "BABEL_LIBDIR" not in os.environ:
        prefix = Path(os.environ.get("CONDA_PREFIX", sys.prefix))
        guess = prefix / "lib" / "openbabel" / "3.1.0"
        if guess.is_dir():
            os.environ["BABEL_LIBDIR"] = str(guess)


def load_labels(casf: Path) -> dict[str, tuple[float, str]]:
    out: dict[str, tuple[float, str]] = {}
    for line in (casf / "power_scoring" / "CoreSet.dat").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        c = line.split()
        if len(c) >= 6:  # noqa: PLR2004
            out[c[0].lower()] = (float(c[3]), c[5])
    return out


def build_model(encoder: str, modpath: str, device: str):  # noqa: ANN201
    """GenScore's own model construction, verbatim."""
    import torch as th  # noqa: PLC0415
    from GenScore.model.model import (  # noqa: PLC0415
        GatedGCN,
        GenScore,
        GraphTransformer,
    )

    enc = GraphTransformer if encoder == "gt" else GatedGCN
    if encoder == "gt":
        kw = {
            "activ_fn": th.nn.SiLU(), "transformer_residual": True,
            "num_attention_heads": 4, "norm_to_apply": "batch",
            "dropout_rate": 0.15, "num_layers": 6,
        }
    else:
        kw = {
            "residual": True, "dropout_rate": 0.15,
            "equivstable_pe": False, "num_layers": 6,
        }
    lig = enc(in_channels=MP["num_node_featsl"], edge_features=MP["num_edge_featsl"],
              num_hidden_channels=MP["hidden_dim0"], **kw)
    prot = enc(in_channels=MP["num_node_featsp"], edge_features=MP["num_edge_featsp"],
               num_hidden_channels=MP["hidden_dim0"], **kw)
    model = GenScore(lig, prot, in_channels=MP["hidden_dim0"], hidden_dim=MP["hidden_dim"],
                     n_gaussians=MP["n_gaussians"], dropout_rate=MP["dropout_rate"],
                     dist_threhold=MP["dist_threhold"]).to(device)
    model.load_state_dict(
        th.load(modpath, map_location=th.device(device))["model_state_dict"]
    )
    model.eval()
    return model


def get_pocket(pdbid: str, coreset: Path, pockets: Path, extract_pocket, cutoff: float = 10.0) -> str:  # noqa: ANN001
    out = pockets / f"{pdbid}_p_pocket_{cutoff}.pdb"
    if out.is_file():
        return str(out)
    reflig = coreset / pdbid / f"{pdbid}_ligand.sdf"
    if not reflig.is_file():
        reflig = coreset / pdbid / f"{pdbid}_ligand.mol2"
    extract_pocket(str(coreset / pdbid / f"{pdbid}_protein.pdb"), str(reflig), cutoff,
                   protname=f"{pdbid}_p", ligname=f"{pdbid}_l", workdir=str(pockets))
    return str(out)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--casf", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True, help="GenScore submodule root")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint", default="GatedGCN_ft_1.0_1.pth")
    ap.add_argument("--encoder", default="gatedgcn", choices=("gt", "gatedgcn"))
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--repaired-dir", type=Path, default=None,
        help="Inputs for complexes whose distributed files fail graph construction: "
             "<pdbid>_ligand.sdf and/or <pdbid>_pocket.rdkit are used when present.",
    )
    args = ap.parse_args()

    setup_babel()
    sys.path.insert(0, str(args.repo))

    import numpy as np  # noqa: PLC0415
    import torch as th  # noqa: PLC0415
    from GenScore.data.data import VSDataset  # noqa: PLC0415
    from GenScore.feats.extract_pocket_prody import extract_pocket  # noqa: PLC0415
    from GenScore.model.utils import run_an_eval_epoch  # noqa: PLC0415
    from torch_geometric.loader import DataLoader  # noqa: PLC0415

    labels = load_labels(args.casf)
    coreset = args.casf / "coreset"
    pockets = args.work / "pockets"
    sdf_dir = args.work / "crystal_sdf"
    pockets.mkdir(parents=True, exist_ok=True)
    sdf_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if th.cuda.is_available() else "cpu"
    model = build_model(args.encoder,
                        str(args.repo / "trained_models" / args.checkpoint), device)
    print(f"device={device} checkpoint={args.checkpoint} complexes={len(labels)}", flush=True)

    rows, failed = [], []
    for n, (pdbid, (logka, cluster)) in enumerate(sorted(labels.items()), 1):
        try:
            repaired_lig = repaired_prot = None
            if args.repaired_dir is not None:
                cand_lig = args.repaired_dir / f"{pdbid}_ligand.sdf"
                cand_prot = args.repaired_dir / f"{pdbid}_pocket.rdkit"
                repaired_lig = cand_lig if cand_lig.is_file() else None
                repaired_prot = cand_prot if cand_prot.is_file() else None

            if repaired_prot is not None:
                # A PDB round-trip re-infers the bad bond this repair removes,
                # so the corrected pocket has to arrive as a pickled Mol.
                from rdkit import Chem  # noqa: PLC0415

                prot = Chem.Mol(repaired_prot.read_bytes())
            else:
                prot = get_pocket(pdbid, coreset, pockets, extract_pocket)

            if repaired_lig is not None:
                lig = str(repaired_lig)
            else:
                lig_path = sdf_dir / f"{pdbid}_crystal.sdf"
                lig = str(lig_path)
                if not lig_path.is_file():
                    import subprocess  # noqa: PLC0415

                    subprocess.run(  # noqa: S603
                        [os.environ.get("AFFINITY_OBABEL", "obabel"),
                         str(coreset / pdbid / f"{pdbid}_ligand.mol2"), "-O", lig],
                        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
            data = VSDataset(ligs=lig, prot=prot, cutoff=10.0,
                             explicit_H=False, use_chirality=True, parallel=False)
            loader = DataLoader(dataset=data, batch_size=MP["batch_size"],
                                shuffle=False, num_workers=args.num_workers)
            preds = run_an_eval_epoch(model, loader, pred=True,
                                      dist_threhold=MP["dist_threhold"], device=device)
            score = float(np.asarray(preds, dtype=float).ravel()[0])
        except Exception as exc:  # noqa: BLE001
            failed.append((pdbid, repr(exc)[:120]))
            print(f"[{n}/{len(labels)}] {pdbid} FAILED: {exc!r:.120}", flush=True)
            continue
        rows.append({"pdbid": pdbid, "logka": logka, "cluster": cluster, "head": score})
        if n % 25 == 0:
            print(f"[{n}/{len(labels)}] {pdbid} score={score:.3f}", flush=True)

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
    if failed:
        print(f"FAILED {len(failed)}: {failed[:10]}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

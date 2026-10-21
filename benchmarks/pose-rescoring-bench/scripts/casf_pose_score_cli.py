"""Score the CASF-2016 docking-power decoy stacks with RTMScore or GenScore.

Runs inside the chosen baseline's environment (DGL for RTMScore,
PyTorch-Geometric for GenScore), driven by
``pose_rescoring_bench.adapters.{rtmscore,genscore}``.

For every docking-power target it

1. extracts the 10 A pocket from ``<pdbid>_protein.pdb`` around the native
   ligand with the ProDy routine both repositories ship, cached on disk;
2. converts the ``<pdbid>_decoys.mol2`` stack to sdf with OpenBabel -- RDKit's
   mol2 reader fails silently on some CASF files -- cached;
3. scores every pose;
4. joins the provided ``<pdbid>_rmsd.dat`` and writes this bench's per-pose dump
   (``pdbid,pose,rmsd,head``), plus the authors' per-target ``<pdbid>_score.dat``
   so their own ``scripts/casf/docking_power2.py`` can consume it verbatim.

The model construction, hyperparameters and pocket extraction are the upstream
repositories' own; this file is the glue. The hyperparameters below are copied
verbatim -- they are not defaults, they define the released models'
architectures, and a mismatch loads weights into the wrong shape. Note that the
two backends differ in the head's dropout rate (0.10 against 0.15), which is
upstream's choice and not a typo here.

Higher score = more native-like for both backends.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

#: Shared architecture. Verbatim from the upstream CASF harnesses.
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

#: RTMScore's released head uses a lower dropout than GenScore's.
_RTMSCORE_DROPOUT = 0.10

#: Pocket radius around the native ligand, in angstrom. Upstream's value.
_POCKET_CUTOFF = 10.0


def setup_babel() -> None:
    """Point OpenBabel at its data directory when the env has not."""
    if "BABEL_LIBDIR" not in os.environ:
        prefix = Path(os.environ.get("CONDA_PREFIX", sys.prefix))
        guess = prefix / "lib" / "openbabel" / "3.1.0"
        if guess.is_dir():
            os.environ["BABEL_LIBDIR"] = str(guess)


def load_targets(casf: Path) -> list[str]:
    """The docking-power target list.

    Distinct from ``power_scoring/CoreSet.dat``: docking power is defined over
    the targets that ship a decoy stack.
    """
    import pandas as pd  # noqa: PLC0415

    core = pd.read_csv(
        casf / "power_docking" / "CoreSet.dat", sep=r"[,\t ]+", engine="python"
    )
    return [str(x) for x in core["#code"]]


def build_backend(backend: str, repo: Path):  # noqa: ANN201
    """Import a repository and return its (VSDataset, loader, model, pocket, eval)."""
    sys.path.insert(0, str(repo))
    import torch as th  # noqa: PLC0415

    if backend == "genscore":
        from GenScore.data.data import VSDataset  # noqa: PLC0415
        from GenScore.feats.extract_pocket_prody import extract_pocket  # noqa: PLC0415
        from GenScore.model.model import (  # noqa: PLC0415
            GatedGCN,
            GenScore,
            GraphTransformer,
        )
        from GenScore.model.utils import run_an_eval_epoch  # noqa: PLC0415
        from torch_geometric.loader import DataLoader  # noqa: PLC0415

        def make_loader(data, num_workers):  # noqa: ANN001, ANN202
            return DataLoader(
                dataset=data, batch_size=MP["batch_size"],
                shuffle=False, num_workers=num_workers,
            )

        def build_model(encoder, modpath, device):  # noqa: ANN001, ANN202
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
            lig = enc(
                in_channels=MP["num_node_featsl"],
                edge_features=MP["num_edge_featsl"],
                num_hidden_channels=MP["hidden_dim0"], **kw,
            )
            prot = enc(
                in_channels=MP["num_node_featsp"],
                edge_features=MP["num_edge_featsp"],
                num_hidden_channels=MP["hidden_dim0"], **kw,
            )
            model = GenScore(
                lig, prot, in_channels=MP["hidden_dim0"],
                hidden_dim=MP["hidden_dim"], n_gaussians=MP["n_gaussians"],
                dropout_rate=MP["dropout_rate"],
                dist_threhold=MP["dist_threhold"],
            ).to(device)
            model.load_state_dict(
                th.load(modpath, map_location=th.device(device))["model_state_dict"]
            )
            model.eval()
            return model

        return VSDataset, make_loader, build_model, extract_pocket, run_an_eval_epoch

    if backend == "rtmscore":
        from RTMScore.data.data import VSDataset  # noqa: PLC0415
        from RTMScore.feats.extract_pocket_prody import extract_pocket  # noqa: PLC0415
        from RTMScore.model.model2 import (  # noqa: PLC0415
            DGLGraphTransformer,
            RTMScore,
        )
        from RTMScore.model.utils import collate, run_an_eval_epoch  # noqa: PLC0415
        from torch.utils.data import DataLoader  # noqa: PLC0415

        def make_loader(data, num_workers):  # noqa: ANN001, ANN202
            return DataLoader(
                dataset=data, batch_size=MP["batch_size"], shuffle=False,
                num_workers=num_workers, collate_fn=collate,
            )

        def build_model(encoder, modpath, device):  # noqa: ANN001, ANN202, ARG001
            kw = {
                "activ_fn": th.nn.SiLU(), "transformer_residual": True,
                "num_attention_heads": 4, "norm_to_apply": "batch",
                "dropout_rate": 0.15, "num_layers": 6,
            }
            lig = DGLGraphTransformer(
                in_channels=MP["num_node_featsl"],
                edge_features=MP["num_edge_featsl"],
                num_hidden_channels=MP["hidden_dim0"], **kw,
            )
            prot = DGLGraphTransformer(
                in_channels=MP["num_node_featsp"],
                edge_features=MP["num_edge_featsp"],
                num_hidden_channels=MP["hidden_dim0"], **kw,
            )
            model = RTMScore(
                lig, prot, in_channels=MP["hidden_dim0"],
                hidden_dim=MP["hidden_dim"], n_gaussians=MP["n_gaussians"],
                dropout_rate=_RTMSCORE_DROPOUT,
                dist_threhold=MP["dist_threhold"],
            ).to(device)
            model.load_state_dict(
                th.load(modpath, map_location=th.device(device))["model_state_dict"]
            )
            model.eval()
            return model

        return VSDataset, make_loader, build_model, extract_pocket, run_an_eval_epoch

    msg = f"backend must be 'genscore' or 'rtmscore', got {backend!r}"
    raise ValueError(msg)


def get_pocket(pdbid: str, coreset: Path, pockets: Path, extract_pocket) -> str:  # noqa: ANN001
    out = pockets / f"{pdbid}_p_pocket_{_POCKET_CUTOFF}.pdb"
    if out.is_file():
        return str(out)
    reflig = coreset / pdbid / f"{pdbid}_ligand.sdf"
    if not reflig.is_file():
        reflig = coreset / pdbid / f"{pdbid}_ligand.mol2"
    extract_pocket(
        str(coreset / pdbid / f"{pdbid}_protein.pdb"), str(reflig), _POCKET_CUTOFF,
        protname=f"{pdbid}_p", ligname=f"{pdbid}_l", workdir=str(pockets),
    )
    return str(out)


def get_decoys_sdf(pdbid: str, decoys_mol2: Path, sdf_dir: Path, obabel: str) -> str:
    out = sdf_dir / f"{pdbid}_decoys.sdf"
    if out.is_file() and out.stat().st_size > 0:
        return str(out)
    subprocess.run(  # noqa: S603
        [obabel, str(decoys_mol2), "-O", str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return str(out)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True, choices=("genscore", "rtmscore"))
    ap.add_argument("--repo", type=Path, required=True, help="submodule root")
    ap.add_argument("--casf", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint", required=True, help="filename in trained_models/")
    ap.add_argument("--encoder", default="gatedgcn", choices=("gt", "gatedgcn"),
                    help="GenScore only; must match the checkpoint")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--obabel", default=os.environ.get("RESCORING_BENCH_OBABEL",
                                                       "obabel"))
    ap.add_argument("--pdbids", default=None, help="comma-separated subset")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    setup_babel()

    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    import torch as th  # noqa: PLC0415

    vs_dataset, make_loader, build_model, extract_pocket, run_eval = build_backend(
        args.backend, args.repo
    )

    coreset = args.casf / "coreset"
    decoys_dir = args.casf / "decoys_docking"
    pockets = args.work / "pockets"
    sdf_dir = args.work / "decoys_sdf"
    score_dir = args.work / "scores"
    for d in (pockets, sdf_dir, score_dir):
        d.mkdir(parents=True, exist_ok=True)

    targets = args.pdbids.split(",") if args.pdbids else load_targets(args.casf)
    device = "cuda" if th.cuda.is_available() else "cpu"
    model = build_model(
        args.encoder, str(args.repo / "trained_models" / args.checkpoint), device
    )
    print(
        f"backend={args.backend} device={device} checkpoint={args.checkpoint} "
        f"targets={len(targets)}",
        flush=True,
    )

    rows, failed = [], []
    for n, pdbid in enumerate(targets, 1):
        try:
            pocket = get_pocket(pdbid, coreset, pockets, extract_pocket)
            sdf = get_decoys_sdf(
                pdbid, decoys_dir / f"{pdbid}_decoys.mol2", sdf_dir, args.obabel
            )
            data = vs_dataset(
                ligs=sdf, prot=pocket, cutoff=_POCKET_CUTOFF,
                explicit_H=False, use_chirality=True, parallel=True,
            )
            loader = make_loader(data, args.num_workers)
            preds = run_eval(
                model, loader, pred=True,
                dist_threhold=MP["dist_threhold"], device=device,
            )
            # VSDataset suffixes a conformer index; the rmsd table has the stem.
            codes = [i.split("-")[0] for i in data.ids]
            scores = np.asarray(preds, dtype=float).ravel()
            df_score = pd.DataFrame({"#code": codes, "score": scores})
            df_score.to_csv(
                score_dir / f"{pdbid}_score.dat", index=False, sep="\t"
            )
            df_rmsd = pd.read_csv(
                decoys_dir / f"{pdbid}_rmsd.dat", sep=r"[, \t]+",
                header=0, engine="python",
            )
            merged = df_rmsd.merge(df_score, on="#code")
            # The native is not a decoy; our arms are scored without it too.
            merged = merged[~merged["#code"].str.contains("_ligand")]
            if merged.empty:
                msg = "no poses survived the rmsd join"
                raise ValueError(msg)  # noqa: TRY301
        except Exception as exc:  # noqa: BLE001
            failed.append((pdbid, repr(exc)[:120]))
            print(f"[{n}/{len(targets)}] {pdbid} FAILED: {exc!r:.120}", flush=True)
            continue

        # No zip(strict=) and no removeprefix here: both baseline environments
        # are Python 3.8, several releases behind the one this bench runs on.
        for rec in merged[["#code", "rmsd", "score"]].to_dict("records"):
            rows.append(  # noqa: PERF401
                {"pdbid": pdbid, "pose": rec["#code"],
                 "rmsd": float(rec["rmsd"]), "head": float(rec["score"])}
            )
        if n % 25 == 0:
            print(f"[{n}/{len(targets)}] {pdbid} n_poses={len(merged)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["pdbid", "pose", "rmsd", "head"])
        writer.writeheader()
        writer.writerows(rows)
    scored = len({r["pdbid"] for r in rows})
    print(
        f"scored {scored}/{len(targets)} targets, {len(rows)} poses -> {args.out}",
        flush=True,
    )
    if failed:
        print(f"FAILED {len(failed)}: {failed[:10]}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

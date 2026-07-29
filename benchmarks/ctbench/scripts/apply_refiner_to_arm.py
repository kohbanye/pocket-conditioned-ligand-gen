"""Apply a trained pose refiner to an arm's poses -- the ML-only counterpart of
``scripts/relax_in_pocket.py``.

The physics relaxation and the learned refiner have so far been measured on
different inputs: the relaxation on a constrained-sampling arm (23.6 heavy atoms)
and the refiner inside generation on raw samples (18.9 atoms). That confounds the
comparison with molecule size. Running the network as a post-processor over the
same arm SDFs makes it apples-to-apples, and the result is a pipeline that needs
no physics at inference -- which is what a fair ``vina_score`` comparison against
DiffSBDD/DiffGui requires.

Optionally projects the network's output back onto the input's bond-length and
bond-angle manifold (``--project``). Coordinate-regression refiners carry ~0.3 A
per-atom error, an order of magnitude above the ~0.05 A tolerance bond lengths
need, which is why the distilled refiner scored better but took PoseBusters
validity from 0.487 to 0.157. The projection uses no pocket information -- it only
restores the molecule's own internal geometry -- so it belongs to the model's
output parametrisation rather than to the scoring pipeline.

Usage (source-repo interpreter)::

    <src>/.venv/bin/python scripts/apply_refiner_to_arm.py \
        --arm sep4096_fin --out-arm fin_mlref --ckpt <refiner.ckpt> --project \
        --targets ABL2_HUMAN_274_551_0 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SOURCE_REPO = Path("/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen")
SBDD_BENCH = Path("/gs/bs/tga-ohuelab/sakano/git/sbdd-bench")
CTBENCH = Path("/gs/bs/tga-ohuelab/sakano/git/complex-tokenizer-bench")
sys.path.insert(0, str(SOURCE_REPO))
sys.path.insert(0, str(CTBENCH))

import torch  # noqa: E402
from prolit.model.pose_refiner import (  # noqa: E402
    PoseRefinerModule,
    refine_ligand_canonical,
)
from rdkit import Chem, RDLogger  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from scripts.build_distill_refine_set import (  # noqa: E402
    ligand_feats_from_mol,
    pocket_context,
)

RDLogger.DisableLog("rdApp.*")


def _pairs_within(mol: Chem.Mol, max_path: int) -> np.ndarray:
    n = mol.GetNumAtoms()
    dm = Chem.GetDistanceMatrix(mol)
    idx = [(i, j) for i in range(n) for j in range(i + 1, n) if 1 <= dm[i, j] <= max_path]
    return np.asarray(idx, dtype=np.int64).reshape(-1, 2)


def project_geometry(
    x: np.ndarray, x_ref: np.ndarray, pairs: np.ndarray, maxiter: int = 200
) -> np.ndarray:
    """Snap ``x`` back to ``x_ref``'s 1-2/1-3 distances, staying as close to x as possible."""
    if pairs.shape[0] == 0:
        return x
    ref = np.linalg.norm(x_ref[pairs[:, 0]] - x_ref[pairs[:, 1]], axis=1)
    n = x.shape[0]

    def fun(flat: np.ndarray) -> tuple[float, np.ndarray]:
        p = flat.reshape(n, 3)
        diff = p[pairs[:, 0]] - p[pairs[:, 1]]
        dist = np.sqrt((diff**2).sum(-1) + 1e-12)
        dev = dist - ref
        e = float((dev**2).sum())
        g = np.zeros_like(p)
        gp = (2.0 * dev / dist)[:, None] * diff
        np.add.at(g, pairs[:, 0], gp)
        np.add.at(g, pairs[:, 1], -gp)
        # weak anchor to the network's own output so the projection is minimal
        d = p - x
        e += 0.01 * float((d**2).sum())
        g += 0.02 * d
        return e, g.ravel()

    res = minimize(fun, x.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    return res.x.reshape(n, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out-arm", required=True)
    ap.add_argument("--ckpt", required=True, help="path relative to the source repo")
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--project", action="store_true")
    ap.add_argument("--project-path", type=int, default=2)
    ap.add_argument("--n-steps", type=int, default=1)
    # The network is an x1-predictor trained on a single generated->relaxed step,
    # and it lands ~0.31 A short of the teacher. Re-feeding its own output lets it
    # take another step toward the same fixed point.
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = (
        PoseRefinerModule.load_from_checkpoint(
            str(SOURCE_REPO / args.ckpt), map_location=device
        )
        .eval()
        .to(device)
    )

    index = json.loads((SBDD_BENCH / "data" / "targets" / "index.json").read_text())
    targets = index["targets"] if isinstance(index, dict) and "targets" in index else index
    by_id = {t["target_id"]: t for t in targets}
    tdir = SBDD_BENCH / "data" / "targets"

    for tid in args.targets:
        meta = by_id.get(tid)
        src = SBDD_BENCH / "outputs" / args.arm / "own" / tid / "generated.sdf"
        if meta is None or not src.exists():
            print(f"[mlref] {tid}: missing", flush=True)
            continue
        ctx = pocket_context(tdir / meta["receptor_pdb"], tdir / meta["ref_ligand_sdf"])
        if ctx is None:
            print(f"[mlref] {tid}: no pocket", flush=True)
            continue
        pkt_x, pkt_feat = ctx
        dst = SBDD_BENCH / "outputs" / args.out_arm / "own" / tid
        dst.mkdir(parents=True, exist_ok=True)
        n_ok = n_in = 0
        with Chem.SDWriter(str(dst / "generated.sdf")) as w:
            for mol in Chem.SDMolSupplier(str(src), sanitize=True, removeHs=True):
                if mol is None or mol.GetNumConformers() == 0:
                    continue
                n_in += 1
                x0 = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
                try:
                    feat = ligand_feats_from_mol(mol)
                    bonds = np.asarray(
                        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()],
                        dtype=np.int64,
                    ).reshape(-1, 2)
                    xr = x0
                    for _ in range(max(1, args.repeat)):
                        xr = refine_ligand_canonical(
                            module, np.asarray(xr, dtype=np.float32), feat,
                            pkt_x, pkt_feat, device=device, bonds=bonds,
                            n_steps=args.n_steps,
                        )
                    if args.project:
                        xr = project_geometry(
                            np.asarray(xr, dtype=np.float64),
                            np.asarray(x0, dtype=np.float64),
                            _pairs_within(mol, args.project_path),
                        )
                    out = Chem.Mol(mol)
                    oc = out.GetConformer()
                    for i, (px, py, pz) in enumerate(np.asarray(xr)):
                        oc.SetAtomPosition(i, (float(px), float(py), float(pz)))
                    w.write(out)
                    n_ok += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"[mlref] {tid} mol skipped: {exc!r}", flush=True)
                    w.write(mol)
        print(f"[mlref] {tid}: {n_in} in, {n_ok} refined", flush=True)


if __name__ == "__main__":
    main()

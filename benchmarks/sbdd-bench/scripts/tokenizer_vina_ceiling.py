"""What does the tokenizer alone cost, in kcal/mol?

The quantizer reproduces a reference ligand's coordinates to a median 0.344 A.
The pose-precision curve says a 0.40 A mean displacement is already worth about
2 kcal/mol of Vina Score. If that transfers, then no language model on top of
this codebook can reach a good score, however well it predicts -- the ceiling
is in the representation, not the predictor.

So measure it directly rather than inferring it: take each reference ligand,
encode it and decode it back through the VQ-VAE, and score the round-tripped
pose the same way every generated pose is scored. The molecule is unchanged --
same atoms, same bonds -- so the only difference is quantization.

This is the number that decides whether to work on the language model or on the
tokenizer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

BENCH = Path(__file__).resolve().parent.parent
REPO = BENCH.parent.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from generate_ligands_3d import load_atom_norm_stats, load_atom_vqvae  # noqa: E402
from prolit.api import (  # noqa: E402
    ATOM_LAYOUT,
    LigandAtomDescriptor,
    compute_canonical_frame,
    extract_pocket_atoms_from_candidates,
    fields_by_name,
    infer_bonds,
    parse_sdf,
    precompute_pocket_atom_candidates,
)
from prolit.config import PocketExtractionConfig  # noqa: E402
from prolit.tokenizers.atom import spherical_to_cartesian_np  # noqa: E402

from sbdd_bench import datasets, docking, molio  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vqvae-ckpt", required=True)
    p.add_argument("--norm-stats", required=True)
    p.add_argument("--codebook-size", type=int, default=8192)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--shard", default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vq = load_atom_vqvae(a.vqvae_ckpt, a.codebook_size, dev)
    ns = load_atom_norm_stats(a.norm_stats, dev)
    cfg = PocketExtractionConfig()
    ldesc = LigandAtomDescriptor()
    cf = fields_by_name(ATOM_LAYOUT)["coord"]
    cm, cs = ns["atom_mean"][cf.start : cf.end], ns["atom_std"][cf.start : cf.end]

    targets = datasets.load_targets()[: a.limit]
    if a.shard:
        k, n = (int(v) for v in a.shard.split("/"))
        targets = targets[k::n]

    rows = []
    for t in targets:
        if not t.receptor_pdbqt:
            continue
        try:
            mols = parse_sdf(Path(t.ref_ligand_sdf))
            if not mols:
                continue
            mol = mols[0]
            heavy_idx = [i for i, x in enumerate(mol["atoms"]) if x[0] != "H"]
            heavy = np.array(
                [(mol["atoms"][i][1], mol["atoms"][i][2], mol["atoms"][i][3])
                 for i in heavy_idx], dtype=np.float32)
            if len(heavy) < 5:
                continue
            pre = precompute_pocket_atom_candidates(Path(t.receptor_pdb))
            pocket = extract_pocket_atoms_from_candidates(pre, heavy, cfg)
            if pocket is None or pocket.atom_coords.shape[0] == 0:
                continue
            frame = compute_canonical_frame(pocket.ca_coords.astype(np.float64))
            centroid, rot = frame
            bonds = mol.get("bonds") or infer_bonds(
                [x[0] for x in mol["atoms"]],
                np.array([(x[1], x[2], x[3]) for x in mol["atoms"]]))
            larr, elems, meta = ldesc.compute(mol["atoms"], bonds, frame)
            with torch.no_grad():
                lt = (torch.from_numpy(larr).to(dev) - ns["atom_mean"]) / ns["atom_std"]
                codes = vq.encode(lt)
                out = vq.decode_to_outputs(codes)
                c = (out["coord"] * cs + cm).detach().cpu().numpy()
            canon = spherical_to_cartesian_np(c)
            rt = canon @ rot + centroid                      # back to world frame

            order = meta["heavy_to_orig"]
            orig = np.array([(mol["atoms"][i][1], mol["atoms"][i][2],
                              mol["atoms"][i][3]) for i in order], dtype=np.float64)
            els = [mol["atoms"][i][0] for i in order]
            rmsd = float(np.sqrt(((rt - orig) ** 2).sum(1).mean()))
            disp = float(np.linalg.norm(rt - orig, axis=1).mean())
            # Split the damage: a pose can be displaced as a rigid whole (which
            # Vina forgives, and which local optimisation recovers) or have its
            # bonds stretched (which Vina charges for directly). The fix is a
            # different one in each case, so they must not be reported as one
            # number.
            pos = {o: i for i, o in enumerate(order)}
            pairs = [(pos[u], pos[v]) for u, v, *_ in bonds
                     if u in pos and v in pos]
            if pairs:
                ii = np.array([u for u, _ in pairs])
                jj = np.array([v for _, v in pairs])
                b_o = np.linalg.norm(orig[ii] - orig[jj], axis=1)
                b_r = np.linalg.norm(rt[ii] - rt[jj], axis=1)
                bond_mae = float(np.abs(b_r - b_o).mean())
            else:
                bond_mae = float("nan")
            # rigid part: best-fit superposition residual vs raw displacement
            oc, rc = orig - orig.mean(0), rt - rt.mean(0)
            u_, _, vt = np.linalg.svd(rc.T @ oc)
            dsign = np.sign(np.linalg.det(u_ @ vt))
            rmat = u_ @ np.diag([1.0, 1.0, dsign]) @ vt
            aligned = rc @ rmat
            rmsd_aligned = float(np.sqrt(((aligned - oc) ** 2).sum(1).mean()))
            centroid_shift = float(np.linalg.norm(rt.mean(0) - orig.mean(0)))

            gens = [
                molio.GenMol(idx=0, elements=list(els), coords=orig, tag="crystal"),
                molio.GenMol(idx=1, elements=list(els), coords=rt, tag="roundtrip"),
            ]
            sc = docking.dock_generated(gens, t.receptor_pdbqt, t.box,
                                        modes=("score", "min", "dock"),
                                        workers=min(a.workers, 2), exhaustiveness=8)
            by = {r["idx"]: r for r in sc}
            rows.append({
                "tid": t.target_id, "n_atoms": len(els),
                "rmsd": rmsd, "mean_disp": disp,
                "rmsd_aligned": rmsd_aligned, "centroid_shift": centroid_shift,
                "bond_mae": bond_mae,
                **{f"crystal_{k}": by.get(0, {}).get(f"vina_{k}")
                   for k in ("score", "min", "dock")},
                **{f"rt_{k}": by.get(1, {}).get(f"vina_{k}")
                   for k in ("score", "min", "dock")},
            })
            print(f"{t.target_id[:28]:28s} rmsd {rmsd:.3f} "
                  f"crystal {by.get(0,{}).get('vina_score')} "
                  f"-> roundtrip {by.get(1,{}).get('vina_score')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {t.target_id}: {exc!r}", flush=True)

    a.out.write_text(json.dumps(rows))
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()

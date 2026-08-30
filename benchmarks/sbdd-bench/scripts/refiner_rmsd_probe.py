"""Does the refiner move the pose TOWARD the crystal, or just off the wall?

``refiner_on_known_error.py`` answers in Vina, which is what the paper reports
but which cannot separate "recovered the pose" from "pushed the ligand out of
the clash". This asks the geometric question directly, and needs no Vina, no
receptor files and no GPU: a pose-refine corpus already holds the deployment
error (``x0``) beside the crystal target (``x1``) in one frame, so the answer is
an RMSD.

It also reports the connectivity a bond perception would read off each pose,
because the refiner is handed ``infer_bonds(x0)`` at generation time and the
Jaccard against the true bonds is 0.31 there -- if refining raises it, feeding
the refiner its own output is worth a second round, and if it does not, it is
not.

Measured for ``refit_press0.6`` on the CLM-corruption corpus: RMSD 2.52 -> 2.70
-> 2.80 over three rounds, Jaccard 0.313 -> 0.291 -> 0.291. That refiner buys
1.4 kcal/mol of Vina on the same poses, so what it does is de-clash, not
recover -- and iterating it makes both worse.

    python benchmarks/sbdd-bench/scripts/refiner_rmsd_probe.py \
        data/pose_refine_clm --refine-ckpt pocket-ligand-refine/_pinned/x.ckpt \
        --limit 200 --rounds 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from prolit.api import LIGAND_ELEMENT_VOCAB
from prolit.chem.pdb_io import infer_bonds
from prolit.model.pose_refiner import (
    FEATURE_FIELDS,
    PoseRefinerModule,
    refine_ligand_canonical,
)


def _load(corpus: Path, split: str) -> dict:
    meta = json.loads((corpus / "meta.json").read_text())
    n_c = meta["splits"][split]["num_complexes"]
    n_r = meta["splits"][split]["num_records"]
    w = len(FEATURE_FIELDS)
    comp = np.fromfile(corpus / f"{split}.complexes", dtype=np.int64).reshape(-1, 3)
    return {
        "comp": comp[:n_c],
        "x1": np.fromfile(corpus / f"{split}.lig_x1", dtype=np.float32).reshape(-1, 3),
        "x0": np.fromfile(corpus / f"{split}.lig_x0", dtype=np.float32).reshape(-1, 3),
        "feat": np.fromfile(corpus / f"{split}.lig_feat", dtype=np.int16).reshape(-1, w),
        "bonds": np.fromfile(corpus / f"{split}.lig_bonds", dtype=np.int32).reshape(-1, 2),
        "pkt": np.fromfile(corpus / f"{split}.pkt_x", dtype=np.float32).reshape(-1, 3),
        "pfeat": np.fromfile(corpus / f"{split}.pkt_feat", dtype=np.int16).reshape(-1, w),
        "rec": np.fromfile(corpus / f"{split}.records", dtype=np.int64)[:n_r],
        "scale": np.fromfile(corpus / f"{split}.record_scale", dtype=np.float32)[:n_r],
    }


def _jaccard(elements: list[str], xyz: np.ndarray, true: set) -> float:
    got = {
        tuple(sorted((int(u), int(v))))
        for u, v, *_ in infer_bonds(elements, np.asarray(xyz, dtype=np.float64))
    }
    return len(true & got) / len(true | got) if (true | got) else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--split", default="val")
    ap.add_argument("--refine-ckpt", type=Path, action="append", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    d = _load(a.corpus, a.split)
    comp = d["comp"]
    o_l = np.concatenate([[0], np.cumsum(comp[:, 0])])
    o_b = np.concatenate([[0], np.cumsum(comp[:, 2])])
    o_p = np.concatenate([[0], np.cumsum(comp[:, 1])])
    elem_col = [n for n, _ in FEATURE_FIELDS].index("element")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The scale-0 records are the decode itself; the graded ones carry extra
    # augmentation that deployment does not apply.
    picks, pos = [], 0
    for i, cid in enumerate(d["rec"]):
        n = int(comp[cid, 0])
        if d["scale"][i] == 0.0:
            picks.append((int(cid), d["x0"][pos : pos + n]))
        pos += n
    picks = picks[: a.limit]

    for ckpt in a.refine_ckpt:
        net = (
            PoseRefinerModule.load_from_checkpoint(ckpt, map_location=dev)
            .eval()
            .to(dev)
        )
        rows = []
        for cid, x0 in picks:
            n = int(comp[cid, 0])
            els = [
                LIGAND_ELEMENT_VOCAB[k] if LIGAND_ELEMENT_VOCAB[k] != "OTHER" else "C"
                for k in d["feat"][o_l[cid] : o_l[cid] + n, elem_col]
            ]
            true = {
                tuple(sorted(t))
                for t in d["bonds"][o_b[cid] : o_b[cid] + comp[cid, 2]].tolist()
            }
            tgt = d["x1"][o_l[cid] : o_l[cid] + n]
            lf = d["feat"][o_l[cid] : o_l[cid] + n]
            pk = d["pkt"][o_p[cid] : o_p[cid] + comp[cid, 1]]
            pf = d["pfeat"][o_p[cid] : o_p[cid] + comp[cid, 1]]
            cur = np.asarray(x0, dtype=np.float32)
            r = [float(np.sqrt(((cur - tgt) ** 2).sum(1).mean()))]
            j = [_jaccard(els, cur, true)]
            for _ in range(a.rounds):
                b = np.asarray(
                    infer_bonds(els, cur.astype(np.float64)), dtype=np.int64
                ).reshape(-1, 2)
                cur = np.asarray(
                    refine_ligand_canonical(
                        net, cur, lf, pk, pf, bonds=b, device=dev
                    ),
                    dtype=np.float32,
                )
                r.append(float(np.sqrt(((cur - tgt) ** 2).sum(1).mean())))
                j.append(_jaccard(els, cur, true))
            rows.append(r + j)
        arr = np.array(rows)
        n_r = a.rounds + 1
        print(f"\n=== {ckpt.name}  ({len(arr)} complexes, {a.split}) ===")
        print(f"{'round':>6s} {'RMSD':>7s} {'改善':>7s} {'bond Jaccard':>13s}")
        for k in range(n_r):
            gain = np.median(arr[:, k] - arr[:, 0])
            print(
                f"{k:6d} {np.median(arr[:, k]):7.3f} {gain:+7.3f}"
                f" {np.median(arr[:, n_r + k]):13.3f}"
            )
        best = arr[:, 1] - arr[:, 0]
        print(f"  1 round で改善した錯体 {(best < 0).mean():.1%}")


if __name__ == "__main__":
    main()

"""凹面重み付き重心に近い分子を選ぶと、Vina は良くなるか。

単純なポケット重心 (推定誤差 2.26 A) では効かなかった。凹面重み付き
重心は 1.83 A と良い推定量なので、選別基準としても効くはず。

使うのはポケットの座標だけ。参照リガンドも Vina も使わない。
FLOWR も同じポケットを --cut_pocket で受け取る。
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "benchmarks/sbdd-bench")
from scipy import stats  # noqa: E402

from sbdd_bench import datasets, molio  # noqa: E402
from sbdd_bench import pose as bpose  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=None)
A = ap.parse_args()
ts = {x.target_id: x for x in datasets.load_targets()}
OUT = Path("benchmarks/sbdd-bench/outputs/gen100_arom_post/own")
rows = []
for p in sorted(OUT.glob("*/generated.sdf")):
    tid = p.parent.name
    tg = ts.get(tid)
    if tg is None or not tg.pocket_pdb or not Path(tg.pocket_pdb).exists():
        continue
    _, P = bpose.read_protein_heavy(tg.pocket_pdb)
    _, REC = bpose.read_protein_heavy(tg.receptor_pdb)
    if len(P) < 20:
        continue
    # Burial counted against nearby receptor atoms only -- the full receptor is
    # tens of thousands of atoms and the count past 8 A is zero anyway.
    near = REC[np.linalg.norm(REC - P.mean(0), axis=1) < 25.0]
    nb = (np.linalg.norm(P[:, None, :] - near[None, :, :], axis=2) < 8.0).sum(1)
    w = 1.0 / (1.0 + nb.astype(float))
    concave = (P * w[:, None]).sum(0) / w.sum()
    plain = P.mean(0)
    try:
        gens = [g for g in molio.load_generated(p) if g.tag != "ref" and g.mol is not None]
    except OSError:
        continue
    for g in gens:
        c = np.asarray(g.mol.GetConformer().GetPositions(), dtype=np.float64).mean(0)
        rows.append({"target_id": tid, "idx": g.idx,
                     "d_concave": float(np.linalg.norm(c - concave)),
                     "d_plain": float(np.linalg.norm(c - plain))})
F = pd.DataFrame(rows)
M = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(
    'benchmarks/sbdd-bench/results_gen100_arom_post_s*/per_molecule.parquet'))],
    ignore_index=True)
for c in ("vina_score","vina_min","vina_dock"):
    M.loc[M[c]==0, c] = np.nan
d = M.merge(F, on=["target_id","idx"]).dropna(subset=["vina_score","vina_min","vina_dock"])
print(f"分子 {len(d)} / 標的 {d.target_id.nunique()}\n")
for c in ("d_concave","d_plain"):
    r = [stats.spearmanr(g[c], g.vina_score).statistic for _, g in d.groupby("target_id")
         if g[c].nunique() > 3]
    r = np.array([x for x in r if np.isfinite(x)])
    print(f"  標的内 Spearman({c}, vina_score) 中央 {np.median(r):+.3f}  正 {np.mean(r>0):.0%}")
print(f"\n{'選別':30s} {'n':>6s} {'PB':>7s} {'score':>8s} {'min':>8s} {'dock':>8s}")
print(f"{'(選別なし)':30s} {len(d):6d} {d.pb_valid.mean():7.3f} {d.vina_score.median():8.2f} "
      f"{d.vina_min.median():8.2f} {d.vina_dock.median():8.2f}")
for key, lab in (("d_concave","凹面重心に近い"), ("d_plain","素の重心に近い")):
    for frac in (0.25, 0.10):
        k = pd.concat([g.nsmallest(max(1,int(round(len(g)*frac))), key)
                       for _, g in d.groupby("target_id")])
        print(f"{lab+f' 上位{frac:.0%}':30s} {len(k):6d} {k.pb_valid.mean():7.3f} "
              f"{k.vina_score.median():8.2f} {k.vina_min.median():8.2f} {k.vina_dock.median():8.2f}")

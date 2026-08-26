"""The ML-only table: ProLIT arms against FLOWR, paired by target.

Every ProLIT number the project reported before 2026-08-24 had passed through
two numerical optimisers (``rigid_pocket_fit`` and an MMFF/torsion relaxation).
FLOWR's are a network's output. Comparing them is not a comparison, so this
builds the table from generation trees that ran no optimiser, and states the
paired test rather than a difference of medians.

Pairing is by target: models are compared only on targets both produced
molecules for, and the statistic is the median of the per-target differences
with a Wilcoxon signed-rank p. Reporting the difference of two medians over
different target sets is how the same mistake gets made again.

    python benchmarks/sbdd-bench/scripts/ml_only_table.py \
        --arm "ProLIT (press0.6)=press_ml_s*" \
        --arm "ProLIT (deploy)=mlonly_s*" \
        --baseline "FLOWR=flowr100_s*"
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BENCH = Path(__file__).resolve().parent.parent
METRICS = ("vina_score", "vina_min", "vina_dock")


def load(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(BENCH / f"results_{pattern}" / "per_molecule.parquet")))
    if not files:
        msg = f"no per_molecule.parquet under results_{pattern}"
        raise SystemExit(msg)
    m = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    for c in METRICS:
        # Vina writes 0.0 where it wrote nothing; a real score is never exactly 0.
        m.loc[m[c] == 0, c] = np.nan
    return m.dropna(subset=list(METRICS)).copy()


def per_target(m: pd.DataFrame) -> pd.DataFrame:
    g = m.groupby("target_id")
    return pd.DataFrame({
        "score": g.vina_score.median(),
        "min": g.vina_min.median(),
        "dock": g.vina_dock.median(),
        "clash_free": g.clash_count.apply(lambda s: (s == 0).mean()),
        "pb": g.pb_valid.mean(),
        "strain": g.strain_energy.median(),
        "min_rmsd": g.min_rmsd.median(),
        "n_mol": g.size(),
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[], metavar="LABEL=GLOB")
    ap.add_argument("--baseline", required=True, metavar="LABEL=GLOB")
    a = ap.parse_args()

    blabel, bglob = a.baseline.split("=", 1)
    base = per_target(load(bglob))
    arms = {}
    for spec in a.arm:
        label, pat = spec.split("=", 1)
        arms[label] = per_target(load(pat))

    print(f"{'アーム':28s} {'標的':>4s} {'score':>7s} {'min':>7s} {'dock':>7s}"
          f" {'clash-free':>10s} {'PB':>6s} {'strain':>7s} {'rmsd':>6s}")
    for label, t in [*arms.items(), (blabel, base)]:
        print(f"{label[:28]:28s} {len(t):4d} {t.score.median():7.2f} {t['min'].median():7.2f}"
              f" {t.dock.median():7.2f} {t.clash_free.mean():10.3f} {t.pb.mean():6.3f}"
              f" {t.strain.median():7.1f} {t.min_rmsd.median():6.2f}")

    for label, t in arms.items():
        print(f"\n=== {label} vs {blabel} (共通標的で対応をとる) ===")
        j = t.join(base, how="inner", lsuffix="_a", rsuffix="_b")
        print(f"  共通標的 {len(j)}")
        for k in ("score", "min", "dock"):
            d = j[f"{k}_a"] - j[f"{k}_b"]
            p = stats.wilcoxon(d).pvalue if len(d) > 5 else float("nan")
            print(f"  {k:6s} 差の中央値 {d.median():+7.2f}   "
                  f"{label} が良い標的 {(d < 0).mean():5.1%}   p={p:.2g}")

    labels = list(arms)
    for i, la in enumerate(labels):
        for lb in labels[i + 1 :]:
            print(f"\n=== {la} vs {lb} (アーム同士) ===")
            j = arms[la].join(arms[lb], how="inner", lsuffix="_a", rsuffix="_b")
            for k in ("score", "min", "dock"):
                d = j[f"{k}_a"] - j[f"{k}_b"]
                p = stats.wilcoxon(d).pvalue if len(d) > 5 else float("nan")
                print(f"  {k:6s} 差の中央値 {d.median():+7.2f}   "
                      f"{la} が良い標的 {(d < 0).mean():5.1%}   p={p:.2g}")


if __name__ == "__main__":
    main()

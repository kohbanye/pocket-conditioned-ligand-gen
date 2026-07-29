"""Paired per-target comparison of generation arms (ours vs ours, ours vs baseline).

Reads ``results/generation/<arm>/per_molecule.parquet`` (the sbdd-bench dump) for
each arm named on the command line plus the DiffSBDD reference dump, aggregates
per target, and reports paired tests on the metrics the paper argues about:
``vina_dock`` (headline), plus validity / connectivity / QED / SA / aromatic
rings / PoseBusters. Answers two questions in one pass:

* does an arm beat DiffSBDD on Vina (the loop's target), and
* does ``joint`` beat ``separate`` (the ablation claim).

Usage::

    uv run python scripts/compare_arms.py joint_bo sep4096_bo --vs-diffsbdd
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path("results/generation")
DIFFSBDD = Path("/gs/bs/tga-ohuelab/sakano/git/sbdd-bench/results/per_molecule.parquet")

METRICS = (
    "vina_dock",
    "vina_min",
    "vina_score",
    "qed",
    "sa",
    "valid",
    "connected",
    "pb_valid",
    "n_atoms",
    "ligand_eff",
)


def load(arm: str) -> pd.DataFrame | None:
    path = RESULTS / arm / "per_molecule.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df[df.get("tag", "") != "ref"].copy()


def load_diffsbdd() -> pd.DataFrame | None:
    if not DIFFSBDD.exists():
        return None
    df = pd.read_parquet(DIFFSBDD)
    col = "model" if "model" in df.columns else None
    if col:
        df = df[df[col].astype(str).str.contains("diffsbdd", case=False)]
    return df[df.get("tag", "") != "ref"].copy()


def per_target(df: pd.DataFrame) -> pd.DataFrame:
    # Ligand efficiency keeps the size lever honest: Vina scores are not
    # size-normalised, so a size-matched comparison needs vina_dock per heavy
    # atom alongside the raw score.
    if {"vina_dock", "n_atoms"} <= set(df.columns):
        df = df.assign(ligand_eff=df["vina_dock"] / df["n_atoms"].clip(lower=1))
    cols = [m for m in METRICS if m in df.columns]
    return df.groupby("target_id")[cols].mean()


def paired(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str) -> None:
    common = a.index.intersection(b.index)
    print(f"\n--- {label_a} vs {label_b}  (paired over {len(common)} targets) ---")
    print(f"{'metric':<12}{label_a:>12}{label_b:>12}{'delta':>10}{'win%':>8}{'p(wilcoxon)':>13}")
    for m in METRICS:
        if m not in a.columns or m not in b.columns:
            continue
        x, y = a.loc[common, m], b.loc[common, m]
        ok = x.notna() & y.notna()
        if ok.sum() < 5:  # noqa: PLR2004
            continue
        x, y = x[ok], y[ok]
        # lower is better for vina_* and sa; higher is better for the rest
        lower_better = m.startswith("vina") or m in {"sa", "ligand_eff"}
        wins = float((x < y).mean() if lower_better else (x > y).mean())
        try:
            p = stats.wilcoxon(x, y).pvalue
        except ValueError:
            p = np.nan
        print(
            f"{m:<12}{x.mean():>12.4f}{y.mean():>12.4f}{x.mean() - y.mean():>10.4f}"
            f"{wins:>8.2f}{p:>13.3g}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--vs-diffsbdd", action="store_true")
    ap.add_argument("--target", type=float, default=-7.35,
                    help="DiffSBDD vina_dock mean to beat (loop goal).")
    args = ap.parse_args()

    loaded: dict[str, pd.DataFrame] = {}
    for arm in args.arms:
        df = load(arm)
        if df is None:
            print(f"[skip] {arm}: no per_molecule.parquet yet")
            continue
        loaded[arm] = df
        vd = df["vina_dock"] if "vina_dock" in df else pd.Series(dtype=float)
        n_ok = int(vd.notna().sum())
        mean = float(vd.mean())
        verdict = "BEATS DiffSBDD" if mean < args.target else "below target"
        print(
            f"[{arm}] n_mol={len(df)} vina_dock n={n_ok} mean={mean:.3f} "
            f"median={float(vd.median()):.3f} frac<-8={float((vd < -8).mean()):.3f} -> {verdict}"
        )

    pts = {k: per_target(v) for k, v in loaded.items()}
    keys = list(pts)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            paired(pts[keys[i]], pts[keys[j]], keys[i], keys[j])

    if args.vs_diffsbdd:
        ref = load_diffsbdd()
        if ref is None:
            print("\n[warn] DiffSBDD dump not found:", DIFFSBDD)
            return
        refpt = per_target(ref)
        for k in keys:
            paired(pts[k], refpt, k, "diffsbdd")


if __name__ == "__main__":
    main()

"""Merge per-model evaluation result dirs into one combined result set.

Each input dir holds ``per_molecule.parquet`` + ``per_target.csv`` written by
``run_evaluation.py``. This concatenates them and recomputes the per-model
aggregate, so an expensive model (e.g. DiffGui) can be evaluated separately and
folded in without re-docking the others.

    python scripts/merge_results.py --inputs results results/diffgui --out results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def per_model_table(per_target: pd.DataFrame) -> pd.DataFrame:
    if per_target.empty:
        return per_target
    num = per_target.select_dtypes("number").columns
    agg = per_target.groupby("model")[list(num)].mean(numeric_only=True)
    agg["n_targets"] = per_target.groupby("model").size()
    cols = [c for c in [
        "n_targets", "validity", "connected", "pb_valid_rate", "clash_free_rate",
        "qed_mean", "sa_mean", "lipinski_frac", "vina_score_mean", "vina_min_mean",
        "vina_dock_mean", "div_uniqueness", "div_novelty", "div_scaffold_diversity",
        "tanimoto_ref_mean", "hit_rate", "hit_scaffold_unique_rate",
    ] if c in agg.columns]
    return agg[cols].reset_index()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True, help="result dirs to merge")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    mol_frames, tgt_frames = [], []
    for d in args.inputs:
        d = Path(d)
        mp, tp = d / "per_molecule.parquet", d / "per_target.csv"
        if mp.exists():
            mol_frames.append(pd.read_parquet(mp))
        if tp.exists():
            tgt_frames.append(pd.read_csv(tp))

    per_mol = pd.concat(mol_frames, ignore_index=True) if mol_frames else pd.DataFrame()
    per_target = pd.concat(tgt_frames, ignore_index=True) if tgt_frames else pd.DataFrame()
    # De-duplicate (model, target_id[, idx]) in case an input was merged twice.
    if not per_target.empty:
        per_target = per_target.drop_duplicates(["model", "target_id"], keep="last")
    if not per_mol.empty and {"model", "target_id", "idx"}.issubset(per_mol.columns):
        per_mol = per_mol.drop_duplicates(["model", "target_id", "idx"], keep="last")
    per_model = per_model_table(per_target)

    args.out.mkdir(parents=True, exist_ok=True)
    per_mol.to_parquet(args.out / "per_molecule.parquet")
    per_target.to_csv(args.out / "per_target.csv", index=False)
    per_model.to_csv(args.out / "per_model.csv", index=False)
    print(f"merged {len(per_mol)} molecule rows, {len(per_target)} target rows, "
          f"{len(per_model)} models -> {args.out}")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(per_model.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

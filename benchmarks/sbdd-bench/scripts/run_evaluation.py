"""Score every model's generated ligands with the shared evaluation suite.

Runs in the bench env (CPU is fine; docking parallelises across cores). Reads
``outputs/<model>/<target_id>/generated.sdf`` and writes::

    results/per_molecule.parquet   one row per generated ligand, every metric
    results/per_target.csv         one row per (model, target)
    results/per_model.csv          per-model aggregate (mean over targets)

Examples
--------
    # full suite (chem + Vina Score/Min/Dock + PoseBusters) over all targets
    python scripts/run_evaluation.py --models own diffsbdd targetdiff diffgui

    # fast pass: chemistry + pose only, no docking
    python scripts/run_evaluation.py --models own --no-dock

    # cap docking to the first 100 valid mols per target (Vina is slow)
    python scripts/run_evaluation.py --models own --dock-limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sbdd_bench import datasets, diversity, metrics, paths  # noqa: E402


def _load_manifest(model_dir: Path) -> dict[str, str]:
    """Every target with a generated SDF, from all manifests plus the tree.

    A sharded generation run writes ``manifest_K_of_N.json`` per shard, and a
    later single-target rerun writes a plain ``manifest.json`` covering only
    what it did. Reading just one of those hides the rest: a 6-target repair
    run left a ``manifest.json`` with 6 entries beside 94 finished targets, and
    the evaluation reported "no generated SDFs" for all of them.

    So union every manifest, then union the tree itself -- a directory holding
    a generated.sdf is evidence regardless of which run wrote it.
    """
    found: dict[str, str] = {}
    for mpath in sorted(model_dir.glob("manifest*.json")):
        try:
            records = json.loads(mpath.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for r in records:
            if r.get("ok") and r.get("sdf") and Path(r["sdf"]).exists():
                found[r["target_id"]] = r["sdf"]
    for p in model_dir.glob("*/generated.sdf"):
        found.setdefault(p.parent.name, str(p))
    return found


def per_model_table(per_target: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-(model,target) rows into a per-model summary (mean over
    targets), keeping the headline columns."""
    if per_target.empty:
        return per_target
    num = per_target.select_dtypes("number").columns
    agg = per_target.groupby("model")[list(num)].mean(numeric_only=True)
    agg["n_targets"] = per_target.groupby("model").size()
    cols = [c for c in [
        "n_targets", "validity", "connected", "pb_valid_rate", "clash_free_rate",
        "qed_mean", "sa_mean", "lipinski_frac", "vina_score_mean", "vina_min_mean",
        "vina_dock_mean",
        # Medians alongside the means, because the means are not robust here:
        # 7-11% of baseline molecules score positive on Vina and drag the mean
        # by ~13 kcal/mol (TargetDiff reads +7.35 as a mean, -5.60 as a median,
        # against -5.20 published). A table that shows only means reports the
        # outlier rate, not the binding.
        "vina_score_median", "vina_min_median", "vina_dock_median",
        # Strain and the two bond-geometry distances complete the columns FLOWR
        # reports; without them a strained-but-well-scoring model looks good.
        "strain_median", "bond_length_w1", "bond_angle_w1",
        "div_uniqueness", "div_novelty", "div_scaffold_diversity",
        "tanimoto_ref_mean", "hit_rate", "hit_scaffold_unique_rate",
    ] if c in agg.columns]
    return agg[cols].reset_index()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["own"])
    p.add_argument("--index", type=Path, default=datasets.DEFAULT_INDEX)
    p.add_argument("--out-dir", type=Path, default=paths.OUTPUTS_DIR)
    p.add_argument("--results", type=Path, default=paths.RESULTS_DIR)
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument(
        "--shard",
        default=None,
        metavar="K/N",
        help="score only every Nth target, starting at K. Vina Dock is the "
        "slow part (~7 min/target), so a 97-target pass is ~11 h on one node "
        "and ~3 h split four ways. Each shard must write its own --results; "
        "concatenate the parquet/CSV afterwards.",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-dock", action="store_true")
    p.add_argument("--no-pose", action="store_true")
    p.add_argument("--interactions", action="store_true")
    p.add_argument("--dock-modes", nargs="+", default=["score", "min", "dock"])
    p.add_argument("--dock-limit", type=int, default=None)
    p.add_argument("--dock-workers", type=int, default=None)
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--train-smiles", type=Path, default=None,
                   help="canonical-SMILES file for novelty (one per line)")
    args = p.parse_args()

    paths.ensure_dirs()
    selected = datasets.load_targets(args.index, limit=args.limit, ids=args.ids)
    if args.shard is not None:
        k, n = (int(v) for v in args.shard.split("/"))
        if not 0 <= k < n:
            msg = f"--shard {args.shard}: need 0 <= K < N"
            raise SystemExit(msg)
        selected = selected[k::n]
        print(f"[eval] shard {k}/{n}: {len(selected)} targets")
    targets = {t.target_id: t for t in selected}
    train_smiles = diversity.load_train_smiles(args.train_smiles) if args.train_smiles else None
    cfg = metrics.EvalConfig(
        dock=not args.no_dock, pose_quality=not args.no_pose,
        interactions=args.interactions, dock_modes=tuple(args.dock_modes),
        dock_workers=args.dock_workers, dock_exhaustiveness=args.exhaustiveness,
        dock_limit=args.dock_limit, train_smiles=train_smiles,
    )

    per_mol_frames, summaries = [], []
    for model in args.models:
        sdf_by_target = _load_manifest(args.out_dir / model)
        if not sdf_by_target:
            print(f"[eval] {model}: no generated SDFs under {args.out_dir / model}")
            continue
        for tid, sdf in sdf_by_target.items():
            t = targets.get(tid)
            if t is None or not Path(sdf).exists():
                continue
            df, summ = metrics.evaluate_target(model, t, sdf, cfg)
            per_mol_frames.append(df)
            summaries.append(summ)
            print(f"[eval] {model}/{tid}: n={summ.get('n_generated')} "
                  f"valid={summ.get('validity')} pb={summ.get('pb_valid_rate')} "
                  f"vina_dock={summ.get('vina_dock_median')} hit={summ.get('hit_rate')}")

    args.results.mkdir(parents=True, exist_ok=True)
    per_mol = pd.concat(per_mol_frames, ignore_index=True) if per_mol_frames else pd.DataFrame()
    per_target = pd.DataFrame(summaries)
    per_model = per_model_table(per_target)

    per_mol.to_parquet(args.results / "per_molecule.parquet")
    per_target.to_csv(args.results / "per_target.csv", index=False)
    per_model.to_csv(args.results / "per_model.csv", index=False)
    print(f"\n[eval] wrote {len(per_mol)} molecule rows, {len(per_target)} target rows -> {args.results}")
    if not per_model.empty:
        with pd.option_context("display.width", 200, "display.max_columns", 50):
            print("\n=== per-model summary ===")
            print(per_model.to_string(index=False))


if __name__ == "__main__":
    main()

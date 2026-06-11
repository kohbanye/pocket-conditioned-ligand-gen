"""Run reconstruction for the selected models over a dataset; write results.

Examples
--------
# CASP16 held-out complexes. The own model reconstructs pocket+ligand; ESM3 and
# FoldToken reconstruct the same pocket backbones (protein_scope=pocket).
uv run python scripts/run_reconstruction.py \
    --models own_vqvae esm3 foldtoken \
    --dataset casp16 --limit 50 --out results/casp16.parquet

# ESM3/FoldToken on the full CASP proteins (their native scope), own model skipped.
uv run python scripts/run_reconstruction.py \
    --models esm3 foldtoken --dataset casp16 --protein-scope full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from plbench import datasets, paths, runner  # noqa: E402
from plbench.adapters.own_vqvae import OwnVQVAEAdapter  # noqa: E402
from plbench.structio import read_backbone  # noqa: E402


def pocket_keys_from_own(own: OwnVQVAEAdapter) -> dict[str, set]:
    """Map sample_id -> {(chain, resid)} of the pocket residues the own model
    used, read from each ``*_orig_pocket.pdb`` (original author numbering)."""
    keys: dict[str, set] = {}
    for tag, rec in own._records.items():  # noqa: SLF001
        bb = read_backbone(rec.orig_pocket_pdb)
        keys[tag] = {
            (str(c), int(r))
            for c, r in zip(bb.chain_ids, bb.res_ids, strict=False)
        }
    return keys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["own_vqvae", "esm3", "foldtoken"])
    p.add_argument("--dataset", default="casp16")
    p.add_argument("--pdb-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--targets", nargs="*", default=None, help="CASP target ids to keep")
    p.add_argument(
        "--protein-scope",
        choices=["pocket", "full"],
        default="pocket",
        help="how ESM3/FoldToken are scored. Both reconstruct the full CASP "
        "protein (their native task); 'pocket' (default) then restricts the "
        "metrics to the own model's pocket residues for a same-residue, "
        "non-OOD comparison; 'full' scores the whole protein.",
    )
    p.add_argument("--foldtoken-level", type=int, default=8)
    p.add_argument("--own-ckpt", type=Path, default=None)
    p.add_argument("--out", type=Path, default=paths.RESULTS_DIR / "casp16.parquet")
    args = p.parse_args()

    paths.ensure_dirs()
    samples = datasets.build(
        args.dataset, pdb_dir=args.pdb_dir, limit=args.limit, targets=args.targets
    )
    print(f"[plbench] dataset={args.dataset} samples={len(samples)} models={args.models}")

    frames: list[pd.DataFrame] = []
    other_models = [m for m in args.models if m != "own_vqvae"]
    pocket_keys: dict[str, set] | None = None

    # Own model first: it defines the pocket residue subset used to restrict the
    # other models' (full-protein) reconstruction metrics.
    if "own_vqvae" in args.models:
        own = OwnVQVAEAdapter(ckpt=args.own_ckpt)
        own.materialize(samples)
        frames.append(runner.run(["own_vqvae"], samples, prebuilt={"own_vqvae": own}))
        if args.protein_scope == "pocket":
            pocket_keys = pocket_keys_from_own(own)
    elif args.protein_scope == "pocket":
        print("[plbench] protein-scope=pocket needs own_vqvae; scoring full protein.")

    # ESM3 / FoldToken always reconstruct the full CASP protein (native task);
    # pocket_keys restricts the *scoring* to the pocket residues.
    if other_models:
        frames.append(
            runner.run(
                other_models,
                [s for s in samples if s.protein_pdb],
                adapter_kwargs={"foldtoken": {"level": args.foldtoken_level}},
                pocket_keys=pocket_keys,
            )
        )

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    summary = runner.summarize(df)
    summary_csv = args.out.with_suffix(".summary.csv")
    summary.to_csv(summary_csv, index=False)

    print(f"\n[plbench] wrote {len(df)} rows -> {args.out}")
    print(f"[plbench] summary  -> {summary_csv}\n")
    with pd.option_context("display.width", 120):
        print(summary.to_string(index=False))

    failed = df[~df["ok"]] if "ok" in df.columns else df.iloc[:0]
    if len(failed):
        print(f"\n[plbench] {len(failed)} failed reconstructions:")
        print(failed[["model", "sample_id", "error"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

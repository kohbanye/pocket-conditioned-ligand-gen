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
from plbench.adapters.own_allatom import ARMS, OwnAllAtomAdapter  # noqa: E402
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
    p.add_argument(
        "--models", nargs="+",
        default=["own_vqvae", "esm3", "foldtoken", "token_mol"],
    )
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
    p.add_argument("--foldtoken-level", type=int, default=12)
    p.add_argument("--own-ckpt", type=Path, default=None)
    p.add_argument(
        "--allatom-arms",
        nargs="*",
        default=None,
        help="all-atom tokenizer arms to evaluate (default: every arm whose "
        f"weights are trained). Choices: {sorted(ARMS)}",
    )
    p.add_argument(
        "--allatom-min-epoch",
        type=int,
        default=90,
        help="refuse checkpoints below this epoch so a still-training run cannot "
        "silently become a row in the ablation table",
    )
    p.add_argument(
        "--pb-valid",
        action="store_true",
        help="add PoseBusters chemical-validity columns to ligand rows. Slow "
        "(~2 s/molecule: the reconstruction and the crystal reference are both "
        "checked), so pair it with --pb-limit.",
    )
    p.add_argument(
        "--pb-limit",
        type=int,
        default=None,
        help="cap how many samples get PoseBusters checks (default: all). The "
        "pb_valid column then covers fewer samples than the other metrics.",
    )
    p.add_argument(
        "--skip-done-arms",
        action="store_true",
        help="skip arms whose per-arm part file already exists, so a resubmitted "
        "job resumes instead of redoing hours of PoseBusters work",
    )
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
        frames.append(
            runner.run(
                ["own_vqvae"], samples, prebuilt={"own_vqvae": own},
                pb_valid=args.pb_valid, pb_limit=args.pb_limit,
            )
        )
        if args.protein_scope == "pocket":
            pocket_keys = pocket_keys_from_own(own)
    elif args.protein_scope == "pocket":
        print("[plbench] protein-scope=pocket needs own_vqvae; scoring full protein.")

    # All-atom tokenizer arms. Each is its own adapter instance reporting under
    # "own_allatom.<arm>", so they land in the same long table as everything else.
    arms = args.allatom_arms
    if arms is None and "own_allatom" in args.models:
        arms = OwnAllAtomAdapter.ready_arms(args.allatom_min_epoch)
        skipped = sorted(set(ARMS) - set(arms))
        if skipped:
            print(f"[plbench] arms not trained past epoch {args.allatom_min_epoch}: {skipped}")
    # Filter AFTER auto-detection, or --skip-done-arms would silently do nothing
    # in the default (auto-detect) mode, which is exactly when it is needed.
    if args.skip_done_arms and arms:
        keep = []
        for a in arms:
            part = args.out.with_name(f"{args.out.stem}.arm-{a}.parquet")
            if part.exists():
                print(f"[plbench] arm {a}: part file exists, skipping ({part.name})")
            else:
                keep.append(a)
        arms = keep
    for arm in arms or []:
        adapter = OwnAllAtomAdapter(arm=arm, min_epoch=args.allatom_min_epoch)
        print(f"[plbench] all-atom arm {arm}: {ARMS[arm].label}")
        adapter.materialize(samples)
        arm_df = runner.run(
            [adapter.name], samples, prebuilt={adapter.name: adapter},
            pb_valid=args.pb_valid, pb_limit=args.pb_limit,
        )
        # Checkpoint each arm as it lands. Two earlier 10-hour runs were killed
        # at the walltime limit and threw away every completed arm with them; a
        # per-arm part file means a timeout costs only the arm in flight, and a
        # resubmission can skip what is already on disk.
        part = args.out.with_name(f"{args.out.stem}.arm-{arm}.parquet")
        arm_df.to_parquet(part)
        print(f"[plbench] arm {arm}: {len(arm_df)} rows -> {part}")
        frames.append(arm_df)
    other_models = [m for m in other_models if m != "own_allatom"]

    # ESM3 / FoldToken always reconstruct the full CASP protein (native task);
    # pocket_keys restricts the *scoring* to the pocket residues.
    if other_models:
        frames.append(
            runner.run(
                other_models,
                [s for s in samples if s.protein_pdb],
                adapter_kwargs={"foldtoken": {"level": args.foldtoken_level}},
                pocket_keys=pocket_keys,
                pb_valid=args.pb_valid,
                pb_limit=args.pb_limit,
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

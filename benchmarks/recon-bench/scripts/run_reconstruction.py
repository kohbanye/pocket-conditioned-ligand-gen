"""Run reconstruction for the selected models over a dataset; write results.

Examples
--------
# CASP16 held-out complexes. ProLIT reconstructs pocket+ligand; ESM3 and
# FoldToken reconstruct the same pocket backbones (protein_scope=pocket).
uv run python scripts/run_reconstruction.py \
    --models own_allatom esm3 foldtoken \
    --dataset casp16 --limit 50 --out results/casp16.parquet

# ESM3/FoldToken on the full CASP proteins (their native scope), ProLIT skipped.
uv run python scripts/run_reconstruction.py \
    --models esm3 foldtoken --dataset casp16 --protein-scope full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from recon_bench import datasets, paths, runner  # noqa: E402
from recon_bench.adapters.own_allatom import ARMS, OwnAllAtomAdapter  # noqa: E402
from recon_bench.adapters.stapled import StapledAdapter  # noqa: E402


def pocket_keys_from_allatom(adapter: OwnAllAtomAdapter) -> dict[str, set]:
    """Map sample_id -> {(chain, resid)} of the pocket residues ProLIT used.

    ``protein_scope=pocket`` scores the full-protein tokenizers (ESM3,
    FoldToken4, Bio2Token) on the same residues ProLIT sees, so the comparison
    is over one shared region rather than ProLIT's pocket against their whole
    chain. The residue identity comes from the per-atom ``protein_chain`` /
    ``protein_resid`` arrays each arm dumps, in original author numbering.
    """
    keys: dict[str, set] = {}
    for tag, dump in adapter.dumps().items():
        with np.load(dump, allow_pickle=False) as npz:
            chains = npz["protein_chain"]
            resids = npz["protein_resid"]
        keys[tag] = {
            (str(c), int(r)) for c, r in zip(chains, resids, strict=True)
        }
    return keys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models", nargs="+",
        default=["own_allatom", "esm3", "foldtoken", "token_mol"],
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
        "metrics to ProLIT's pocket residues for a same-residue, "
        "non-OOD comparison; 'full' scores the whole protein.",
    )
    p.add_argument("--foldtoken-level", type=int, default=12)
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
    p.add_argument(
        "--stapled-pose-bits",
        nargs="*",
        default=["0", "13", "26", "39", "oracle"],
        help="placement budgets for the `stapled` baseline (ESM3 pocket tokens + "
        "ConfSeq ligand tokens). Neither half carries the ligand's rigid "
        "transform, so the budget is what the concatenation has to be handed "
        "before an interface metric on it means anything; `oracle` is the "
        "unattainable ceiling. One arm per value, reported as "
        "stapled_pose<bits>. Only used when `stapled` is among --models.",
    )
    p.add_argument(
        "--stapled-protein-scope",
        choices=["pocket", "full"],
        default="full",
        help="what ESM3 encodes for the stapled baseline. 'full' (default) is "
        "the whole chain, ESM3's native task, with the pocket residues read out "
        "of it; 'pocket' encodes the pocket alone, which is 7 A worse because a "
        "pocket is discontiguous and ESM3 cannot be told so.",
    )
    p.add_argument(
        "--require-cuda",
        action="store_true",
        help="refuse to start without a visible GPU. ESM3 and the all-atom VQ "
        "both fall back to CPU silently, which turns a 15-minute run into a "
        "multi-hour one that hits its walltime with nothing written -- a whole "
        "node allocation spent to produce no rows. Pass this in every job "
        "script; leave it off for a laptop smoke test.",
    )
    p.add_argument("--out", type=Path, default=paths.RESULTS_DIR / "casp16.parquet")
    args = p.parse_args()

    if args.require_cuda:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            raise SystemExit(
                "[recon_bench] --require-cuda: no GPU visible to torch. "
                "On TSUBAME a MIG slice is not always handed through; ask for "
                "gpu_1 rather than a fractional resource."
            )
        print(f"[recon_bench] cuda: {torch.cuda.get_device_name(0)}")

    paths.ensure_dirs()
    samples = datasets.build(
        args.dataset, pdb_dir=args.pdb_dir, limit=args.limit, targets=args.targets
    )
    print(f"[recon_bench] dataset={args.dataset} samples={len(samples)} models={args.models}")

    frames: list[pd.DataFrame] = []
    other_models = list(args.models)
    pocket_keys: dict[str, set] | None = None

    # All-atom tokenizer arms. Each is its own adapter instance reporting under
    # "own_allatom.<arm>", so they land in the same long table as everything else.
    # These run first: the first arm also defines the pocket residue subset used
    # to restrict the full-protein tokenizers' metrics (protein_scope=pocket).
    arms = args.allatom_arms
    if arms is None and "own_allatom" in args.models:
        arms = OwnAllAtomAdapter.ready_arms(args.allatom_min_epoch)
        skipped = sorted(set(ARMS) - set(arms))
        if skipped:
            print(f"[recon_bench] arms not trained past epoch {args.allatom_min_epoch}: {skipped}")
    # Filter AFTER auto-detection, or --skip-done-arms would silently do nothing
    # in the default (auto-detect) mode, which is exactly when it is needed.
    if args.skip_done_arms and arms:
        keep = []
        for a in arms:
            part = args.out.with_name(f"{args.out.stem}.arm-{a}.parquet")
            if part.exists():
                print(f"[recon_bench] arm {a}: part file exists, skipping ({part.name})")
            else:
                keep.append(a)
        arms = keep
    for arm in arms or []:
        adapter = OwnAllAtomAdapter(arm=arm, min_epoch=args.allatom_min_epoch)
        print(f"[recon_bench] all-atom arm {arm}: {ARMS[arm].label}")
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
        print(f"[recon_bench] arm {arm}: {len(arm_df)} rows -> {part}")
        frames.append(arm_df)
        # Every arm tokenizes the same pocket, so the first one that ran is
        # enough to define the shared residue subset.
        if args.protein_scope == "pocket" and pocket_keys is None:
            pocket_keys = pocket_keys_from_allatom(adapter)
    other_models = [m for m in other_models if m != "own_allatom"]

    # The stapled baseline: one adapter per placement budget, each reporting
    # under its own model name so the budgets land as separate rows of one
    # rate curve rather than as one number with a footnote.
    if "stapled" in other_models:
        other_models = [m for m in other_models if m != "stapled"]
        for spec in args.stapled_pose_bits:
            bits = None if str(spec).lower() == "oracle" else int(spec)
            adapter = StapledAdapter(
                pose_bits=bits, protein_scope=args.stapled_protein_scope
            )
            part = args.out.with_name(f"{args.out.stem}.arm-{adapter.name}.parquet")
            if args.skip_done_arms and part.exists():
                print(f"[recon_bench] {adapter.name}: part file exists, skipping")
                frames.append(pd.read_parquet(part))
                continue
            print(f"[recon_bench] stapled baseline: {adapter.name}")
            arm_df = runner.run(
                [adapter.name],
                [s for s in samples if s.protein_pdb and s.ligand_sdf],
                prebuilt={adapter.name: adapter},
                pb_valid=args.pb_valid,
                pb_limit=args.pb_limit,
            )
            arm_df.to_parquet(part)
            print(f"[recon_bench] {adapter.name}: {len(arm_df)} rows -> {part}")
            frames.append(arm_df)

    if args.protein_scope == "pocket" and pocket_keys is None and other_models:
        print(
            "[recon_bench] protein-scope=pocket needs an own_allatom arm to define "
            "the pocket residues; scoring the full protein instead."
        )

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

    print(f"\n[recon_bench] wrote {len(df)} rows -> {args.out}")
    print(f"[recon_bench] summary  -> {summary_csv}\n")
    with pd.option_context("display.width", 120):
        print(summary.to_string(index=False))

    failed = df[~df["ok"]] if "ok" in df.columns else df.iloc[:0]
    if len(failed):
        print(f"\n[recon_bench] {len(failed)} failed reconstructions:")
        print(failed[["model", "sample_id", "error"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

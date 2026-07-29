"""Run generation for one variant on the CrossDocked2020 100-pocket test set.

This is the standard SBDD benchmark path (the 100 pockets DiffSBDD / TargetDiff /
DiffGui all report on). It drives the sbdd-bench harness end-to-end through the
``own`` adapter, so our joint (``joint``/``joint_nocasf``) and separate
(``separate``/``separate_4096``) tokenizer variants are scored by the *identical*
metric suite / docking protocol as the prior-work baselines.

Flow (GPU for --gen, CPU-parallel Vina for --eval)::

    prepare (once, CPU):
        cd <sbdd-bench>; python scripts/prepare_targets.py \
            --crossdocked-test data/crossdocked_test      # -> data/targets/index.json

    per variant (this script):
        uv run python scripts/infer_generation_crossdocked.py --variant joint_nocasf
        uv run python scripts/infer_generation_crossdocked.py --variant separate_4096

Results land in ``results/generation/<variant>/{per_model,per_target}.csv`` +
``per_molecule.parquet`` — exactly the layout ``ctbench.report`` /
``scripts/make_tables.py`` read for the comparison + ablation tables. Baselines
are collected separately via ``scripts/collect_baselines.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

from ctbench.config import EvalConfig
from ctbench.inference import generation
from ctbench.variants import get

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="separate_4096")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="target index.json (default <sbdd-bench>/data/targets/index.json).",
    )
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--dock-modes", nargs="+", default=["score", "min", "dock"])
    parser.add_argument("--skip-gen", action="store_true", help="reuse existing SDFs.")
    parser.add_argument("--skip-eval", action="store_true", help="generate only.")
    parser.add_argument("--gen-extra", nargs="*", default=None)
    parser.add_argument("--eval-extra", nargs="*", default=None)
    parser.add_argument(
        "--refiner",
        default=None,
        help="Override the variant's pose-refiner checkpoint (path relative to "
        "the source repo). vina_score is scored on the pose as generated, so the "
        "refiner is the lever for it -- unlike vina_dock, which re-docks and "
        "discards our coordinates entirely.",
    )
    parser.add_argument(
        "--out-suffix",
        default="",
        help="suffix appended to the sbdd-bench output dir, so an oversampled "
        "pool can be generated without overwriting the variant's own arm "
        "(e.g. --out-suffix _pool400 -> outputs/<variant>_pool400).",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="restrict generation to these target ids (e.g. the 14 multi-pair "
        "delta); passed through to run_generation --ids.",
    )
    args = parser.parse_args()

    variant = get(args.variant)
    gen = variant.generation
    if gen is not None and args.refiner:
        gen = dataclasses.replace(gen, refiner=args.refiner)
    if gen is None or (gen.vqvae is None and not gen.is_separate):
        logger.error("variant %s has no generation checkpoints", args.variant)
        return

    cfg = EvalConfig()
    default_index = cfg.paths.sbdd_bench_repo / "data" / "targets" / "index.json"
    index = args.index or default_index
    if not index.exists():
        logger.error(
            "target index missing: %s\n"
            "Prepare the 100-pocket set first (in the sbdd-bench repo):\n"
            "  python scripts/prepare_targets.py --crossdocked-test data/crossdocked_test",
            index,
        )
        return

    out_root = cfg.paths.sbdd_bench_repo / "outputs" / (args.variant + args.out_suffix)
    # Absolute: run_evaluation runs with cwd=sbdd-bench, so a relative --results
    # would land under sbdd-bench/ instead of this repo's results/ tree.
    results_dir = (args.results / "generation" / (args.variant + args.out_suffix)).resolve()

    gen_extra = list(args.gen_extra or [])
    if args.ids:
        gen_extra += ["--ids", *args.ids]

    if not args.skip_gen:
        generation.generate_own_crossdocked(
            gen,
            cfg.paths,
            cfg.generation,
            index=index,
            out_root=out_root,
            n_samples=args.n_samples,
            extra_args=gen_extra,
        )
    if not args.skip_eval:
        generation.evaluate_own_crossdocked(
            cfg.paths,
            index=index,
            out_root=out_root,
            results_dir=results_dir,
            dock_modes=tuple(args.dock_modes),
            extra_args=args.eval_extra,
        )
    logger.info("crossdocked generation for %s -> %s", args.variant, results_dir)


if __name__ == "__main__":
    main()

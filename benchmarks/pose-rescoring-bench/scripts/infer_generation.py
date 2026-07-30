"""Run generation inference for one tokenizer variant (generate + sbdd-bench eval).

Drives the source repo's all-atom 3D generator, then the sbdd-bench harness, and
records where outputs landed. Needs a GPU + both sibling repos; run under qsub.
See :mod:`pose_rescoring_bench.inference.generation` for the first-run caveat.

Usage::

    uv run python scripts/infer_generation.py --variant joint
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prolit.seeding import add_seed_argument, seed_from_args

from pose_rescoring_bench.config import EvalConfig
from pose_rescoring_bench.inference import generation
from pose_rescoring_bench.variants import get

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="joint")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Generate only; skip sbdd-bench scoring.",
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        default=None,
        help="Extra args passed to the generator.",
    )
    add_seed_argument(parser)
    args = parser.parse_args()
    seed_from_args(args)

    variant = get(args.variant)
    gen = variant.generation
    if gen is None or (gen.vqvae is None and not gen.is_separate):
        logger.error("variant %s has no generation checkpoints yet", args.variant)
        return
    cfg = EvalConfig()
    out_dir = args.results / "generation" / args.variant / "sdf"
    generation.generate(
        gen,
        cfg.paths,
        cfg.generation,
        out_dir,
        extra_args=args.extra,
    )
    if not args.skip_eval:
        generation.evaluate_with_sbdd(cfg.paths, models=["own"])
    logger.info("generation for %s -> %s", args.variant, out_dir)


if __name__ == "__main__":
    main()

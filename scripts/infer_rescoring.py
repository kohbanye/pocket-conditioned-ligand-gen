"""Run pose-rescoring inference for one tokenizer variant -> per-pose dumps.

Writes ``results/rescoring/<variant>/<head-label>.csv`` (one per pose head).
Needs a GPU + the source repo importable; run under qsub.

Usage::

    uv run python scripts/infer_rescoring.py --variant joint
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ctbench.config import EvalConfig
from ctbench.inference import rescoring
from ctbench.io_dumps import write_pose_scores
from ctbench.variants import get

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="joint")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--max-targets", type=int, default=None)
    args = parser.parse_args()

    variant = get(args.variant)
    rescoring_ckpts = variant.rescoring
    if rescoring_ckpts is None or (
        rescoring_ckpts.vqvae is None and not rescoring_ckpts.is_separate
    ):
        logger.error("variant %s has no rescoring checkpoints yet", args.variant)
        return
    cfg = EvalConfig()
    cfg.rescoring.max_targets = args.max_targets
    out_dir = args.results / "rescoring" / args.variant
    for i, head in enumerate(rescoring_ckpts.heads):
        df = rescoring.score_casf(
            rescoring_ckpts,
            cfg.paths,
            cfg.rescoring,
            head_index=i,
        )
        label = head.label or f"head{i}"
        write_pose_scores(df, out_dir / f"{label}.csv")
        logger.info("wrote %d poses for head %s", len(df), label)


if __name__ == "__main__":
    main()

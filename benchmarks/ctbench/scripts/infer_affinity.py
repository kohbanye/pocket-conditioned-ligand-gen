"""Run affinity inference for one tokenizer variant -> per-complex dumps.

Writes ``results/affinity/<variant>/<head-label>.csv`` (one per ensemble head).
Needs a GPU + the source repo importable; run under qsub.

Usage::

    uv run python scripts/infer_affinity.py --variant joint
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ctbench.config import EvalConfig
from ctbench.inference import affinity
from ctbench.io_dumps import write_affinity
from ctbench.variants import get

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="joint")
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    variant = get(args.variant)
    affinity_ckpts = variant.affinity
    if affinity_ckpts is None or (
        affinity_ckpts.vqvae is None and not affinity_ckpts.is_separate
    ):
        logger.error("variant %s has no affinity checkpoints yet", args.variant)
        return
    cfg = EvalConfig()
    out_dir = args.results / "affinity" / args.variant
    for i, head in enumerate(affinity_ckpts.heads):
        df = affinity.score_casf(
            affinity_ckpts,
            cfg.paths,
            cfg.affinity,
            head_index=i,
        )
        label = head.label or f"head{i}"
        write_affinity(df, out_dir / f"{label}.csv")
        logger.info("wrote %d complexes for head %s", len(df), label)


if __name__ == "__main__":
    main()

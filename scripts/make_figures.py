"""Render comparison + ablation figures from the ``results/`` dump tree.

Analysis-only (no GPU). Writes PNGs to ``results/figures/``.

Usage::

    uv run python scripts/make_figures.py [--results results] [--out results/figures]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ctbench import plotting, report

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results") / "figures")
    args = parser.parse_args()

    # comparison figures (ours vs existing methods)
    aff, _ = report.affinity_comparison(args.results)
    if not aff.empty:
        plotting.bar_comparison(
            aff,
            "scoring_R",
            args.out / "affinity_scoring_R.png",
            ascending=True,
        )
    pose, _ = report.rescoring_comparison(args.results)
    if not pose.empty:
        plotting.bar_comparison(
            pose,
            "DP@2A",
            args.out / "rescoring_DP2.png",
            ascending=True,
        )
    gen, _ = report.generation_comparison(args.results)
    if not gen.empty:
        plotting.bar_comparison(
            gen,
            "vina_score_mean",
            args.out / "generation_vina.png",
            ascending=False,
        )

    # ablation figures (joint vs single-modality) — populated once variants exist
    abl_aff, _ = report.affinity_ablation(args.results)
    if len(abl_aff) > 1:
        plotting.ablation_bars(
            abl_aff,
            ["scoring_R", "ranking_rho"],
            args.out / "ablation_affinity.png",
        )
    abl_pose, _ = report.rescoring_ablation(args.results)
    if len(abl_pose) > 1:
        plotting.ablation_bars(
            abl_pose,
            ["DP@2A", "DP@1A", "ranking_rho"],
            args.out / "ablation_rescoring.png",
        )
    abl_gen, _ = report.generation_ablation(args.results)
    if len(abl_gen) > 1:
        plotting.ablation_bars(
            abl_gen,
            ["vina_score_mean", "vina_min_mean"],
            args.out / "ablation_generation.png",
        )
    logger.info("figures written to %s", args.out)


if __name__ == "__main__":
    main()

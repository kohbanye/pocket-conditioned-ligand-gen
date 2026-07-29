"""Seed ``results/`` with known-reproducible baseline dumps (no recomputation).

Parses the sibling repos' existing per-sample outputs into this repo's canonical
schema and writes them under ``results/``. The reproduction code that could
regenerate these lives in :mod:`ctbench.baselines`; this script only collects.

Usage::

    uv run python scripts/collect_baselines.py [--results results]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ctbench.baselines import casf_affinity, casf_pose, sbdd_gen
from ctbench.config import PathsConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    paths = PathsConfig(results_dir=args.results)

    # affinity baselines
    aff = args.results / "affinity"
    _write(
        casf_affinity.collect_genscore(paths.baselines_repo),
        aff / "genscore" / "scoring.csv",
    )
    _write(casf_affinity.collect_vina(paths.source_repo), aff / "vina" / "scoring.csv")

    # pose baselines (per-pose native_score; rmsd joined at eval time)
    res = args.results / "rescoring"
    for backend in ("rtmscore", "genscore"):
        df = casf_pose.collect_pose_scores(
            casf_pose.default_score_dir(paths.baselines_repo, backend),
        )
        _write(df, res / backend / "pose_scores.csv")

    # generation baselines (sbdd-bench official summaries)
    gen = args.results / "generation" / "baselines"
    _write(sbdd_gen.collect_per_model(paths.sbdd_bench_repo), gen / "per_model.csv")
    _write(sbdd_gen.collect_per_target(paths.sbdd_bench_repo), gen / "per_target.csv")


def _write(df: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)  # type: ignore[attr-defined]
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()

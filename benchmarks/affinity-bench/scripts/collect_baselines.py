"""Seed ``results/`` with the baseline dumps (parse, do not recompute).

Usage::

    uv run python scripts/collect_baselines.py [--results results]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from affinity_bench import baselines
from affinity_bench.config import PathsConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    paths = PathsConfig(results_dir=args.results)

    collectors = (
        ("genscore", lambda: baselines.collect_genscore(paths.baselines_repo)),
        ("vina", lambda: baselines.collect_vina(paths.source_repo)),
        ("boltz2", lambda: baselines.collect_boltz2(paths.source_repo)),
    )
    for name, collect in collectors:
        path = args.results / name / "scoring.csv"
        try:
            df = collect()
        except FileNotFoundError as exc:
            # A baseline whose source dump is not on this machine is skipped,
            # not fatal: the table reports the methods it can actually compare.
            logger.warning("skipping %s (%s)", name, exc)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("wrote %d rows -> %s", len(df), path)


if __name__ == "__main__":
    main()

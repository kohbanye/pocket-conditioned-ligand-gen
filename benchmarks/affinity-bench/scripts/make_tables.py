"""Build the affinity comparison table from the ``results/`` dump tree.

Analysis only -- no GPU, no model inference. Reads per-complex dumps, computes
both metrics and their significance against GenScore, writes CSVs to
``results/tables/`` and prints them, along with the scoreboard that says
whether the target (beat GenScore on BOTH metrics) has been met.

Usage::

    uv run python scripts/make_tables.py [--results results] [--out results/tables]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from affinity_bench import report

if TYPE_CHECKING:
    import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _emit(title: str, table: pd.DataFrame, out_dir: Path, stem: str) -> None:
    if table.empty:
        logger.info("\n## %s\n(no data yet)", title)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.csv"
    table.to_csv(path)
    logger.info("\n## %s  -> %s\n%s", title, path, table.round(4).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results") / "tables")
    args = parser.parse_args()

    # Seed tables first, deliberately. Retraining an arm with nothing changed
    # but the seed moves scoring R by ~0.05 and ranking rho by ~0.04 -- larger
    # than any recipe difference measured so far -- so these are the only two
    # tables that support a claim about one recipe beating another. Printing the
    # single-run table first invited exactly the misreading it caused once.
    spread, paired = report.seed_report(args.results)
    _emit(
        "affinity — per family, mean +/- sd over training seeds  [READ THIS ONE]",
        spread,
        args.out,
        "affinity_seed_spread",
    )
    _emit(
        "affinity — paired by seed (n_better out of n_seeds is the verdict; "
        "a 3-seed p-value has almost no power)",
        paired,
        args.out,
        "affinity_seed_paired",
    )

    metrics, sig = report.comparison(args.results)
    _emit(
        "affinity — per RUN, n=1 each  [a single draw, not a measurement]",
        metrics,
        args.out,
        "affinity_comparison_metrics",
    )
    _emit(
        "affinity — significance vs GenScore, per run (n=1)",
        sig,
        args.out,
        "affinity_comparison_sig",
    )
    _emit(
        "affinity — did we beat GenScore on both? (per run, n=1)",
        report.scoreboard(args.results),
        args.out,
        "affinity_scoreboard",
    )


if __name__ == "__main__":
    main()

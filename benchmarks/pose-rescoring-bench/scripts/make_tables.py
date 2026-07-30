"""Build all comparison and ablation tables from the ``results/`` dump tree.

Runs the analysis layer only (no GPU, no model inference): reads per-sample
dumps, computes metrics + significance for every task, writes CSVs to
``results/tables/`` and prints them. Baseline-comparison tables reproduce the
paper numbers today; ablation tables fill in as single-modality variant dumps
are added under ``results/<task>/<variant>/``.

Usage::

    uv run python scripts/make_tables.py [--results results] [--out results/tables]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pose_rescoring_bench import report

if TYPE_CHECKING:
    import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_TASKS = {
    "affinity": (report.affinity_comparison, report.affinity_ablation),
    "rescoring": (report.rescoring_comparison, report.rescoring_ablation),
    "generation": (report.generation_comparison, report.generation_ablation),
}


def _emit(title: str, table: pd.DataFrame, out_dir: Path, stem: str) -> None:
    if table.empty:
        logger.info("\n## %s\n(no data yet)", title)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.csv"
    table.to_csv(path)
    logger.info("\n## %s  -> %s\n%s", title, path, table.round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results") / "tables")
    args = parser.parse_args()

    for task, (comparison, ablation) in _TASKS.items():
        cmp_metrics, cmp_sig = comparison(args.results)
        _emit(
            f"{task} — comparison (metrics)",
            cmp_metrics,
            args.out,
            f"{task}_comparison_metrics",
        )
        _emit(
            f"{task} — comparison (significance vs reference)",
            cmp_sig,
            args.out,
            f"{task}_comparison_sig",
        )
        abl_metrics, abl_sig = ablation(args.results)
        _emit(
            f"{task} — ablation (metrics)",
            abl_metrics,
            args.out,
            f"{task}_ablation_metrics",
        )
        _emit(
            f"{task} — ablation (significance vs joint)",
            abl_sig,
            args.out,
            f"{task}_ablation_sig",
        )


if __name__ == "__main__":
    main()

"""Paired CASF-2016 comparison: ProLIT against the ESM3 x ConfSeq baseline.

The published table's builders (:func:`report.rescoring_comparison`,
:func:`report.rescoring_ablation`) are wired to fixed arm sets, and this
comparison is neither of them: it is one ProLIT arm against one baseline, on
the targets both could score.

Not every target is in both dumps. ConfSeq fails on 10 of the 285 CASF ligands
outright -- all-or-nothing, 0 of 100 poses rather than some of them, because the
failure is a property of the molecule -- so the baseline scores 275. Both the
McNemar and the Wilcoxon in ``aggregate.rescoring_pairwise`` intersect on target
id before testing, so the paired tests are already on the common set; what this
script adds is saying so out loud, because a reader of the table needs to know
the baseline's n is not the benchmark's n.

Usage::

    uv run python scripts/stapled_significance.py \\
        --prolit e250_div --stapled stapled
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd  # noqa: TC002 -- runtime annotation on _load

from pose_rescoring_bench import aggregate
from pose_rescoring_bench.metrics import rescoring as R
from pose_rescoring_bench.report import _rescoring_variant_heads

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _load(root: Path, name: str) -> pd.DataFrame:
    heads = _rescoring_variant_heads(root / name)
    if not heads:
        msg = f"no scored-pose dumps under {root / name}"
        raise FileNotFoundError(msg)
    return R.zsum(heads)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--prolit", default="e250_div")
    ap.add_argument("--stapled", default="stapled")
    ap.add_argument("--cut", type=float, default=2.0)
    args = ap.parse_args()

    root = args.results / "rescoring"
    scored = {
        "ProLIT": _load(root, args.prolit),
        "ESM3xConfSeq": _load(root, args.stapled),
    }
    for name, df in scored.items():
        logger.info(
            "%s: %d targets, %d poses", name, df["pdbid"].nunique(), len(df)
        )
    common = set.intersection(*(set(d["pdbid"]) for d in scored.values()))
    only_prolit = set(scored["ProLIT"]["pdbid"]) - common
    logger.info(
        "paired on %d common targets; %d scored by ProLIT alone: %s",
        len(common),
        len(only_prolit),
        ", ".join(sorted(only_prolit)) or "-",
    )

    logger.info("\n%s", aggregate.rescoring_metrics(scored).to_string())
    logger.info(
        "\n%s",
        aggregate.rescoring_pairwise(
            scored, reference="ProLIT", cut=args.cut
        ).to_string(),
    )


if __name__ == "__main__":
    main()

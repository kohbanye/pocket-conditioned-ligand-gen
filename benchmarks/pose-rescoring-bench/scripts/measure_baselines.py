"""Measure a pose baseline here, with its own weights, on our decoy stacks.

The counterpart to ``collect_baselines.py``: that script *parses* per-target
outputs a sibling checkout had already produced, which is how the currently
published rows were obtained and which cannot be re-run anywhere that checkout
is absent. This one *runs the model*, from the pinned ``third_party/`` submodule,
and writes the same canonical per-pose dump.

A measured dump is written beside the collected one rather than over it::

    results/rescoring/<name>/pose_scores.csv          collected (published)
    results/rescoring/<name>/pose_scores_measured.csv measured here

Replacing a published row is a decision about the paper, not a side effect of
re-running a script, so the two live side by side until someone makes it.

Usage::

    uv run python scripts/measure_baselines.py --baseline rtmscore
    uv run python scripts/measure_baselines.py --baseline vina --workers 40
    uv run python scripts/measure_baselines.py --baseline genscore \
        --pdbids 1a30,1bcu,1bzc          # a three-target smoke test

The two DGL baselines want a GPU node and their own interpreter (see
``pose_rescoring_bench.paths``); Vina wants a many-core CPU node.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pose_rescoring_bench import adapters, paths
from pose_rescoring_bench.io_dumps import write_pose_scores
from pose_rescoring_bench.metrics import rescoring as R

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, choices=adapters.available())
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--pdbids",
        default=None,
        help="Comma-separated subset; default is the docking-power target list.",
    )
    parser.add_argument("--checkpoint", default=None, help="Override the default.")
    parser.add_argument("--workers", type=int, default=None, help="Vina only.")
    parser.add_argument(
        "--num-workers", type=int, default=None, help="Dataloader workers."
    )
    parser.add_argument(
        "--allow-missing",
        default="",
        help="Comma-separated targets this method provably cannot process, so "
        "a shortfall there is expected rather than an error. GenScore and "
        "RTMScore both fail to build a pocket graph for 4k18 from the "
        "distributed inputs, and their published numbers are on the other 284. "
        "Whatever is passed here is logged beside the result.",
    )
    args = parser.parse_args()

    kwargs: dict[str, object] = {}
    if args.checkpoint is not None:
        kwargs["checkpoint"] = args.checkpoint
    if args.workers is not None:
        kwargs["workers"] = args.workers
    if args.num_workers is not None:
        kwargs["num_workers"] = args.num_workers
    allow = tuple(x for x in args.allow_missing.split(",") if x)
    if allow:
        kwargs["allow_missing"] = allow
    baseline = adapters.build(args.baseline, **kwargs)

    targets = (
        args.pdbids.split(",")
        if args.pdbids
        else _docking_targets(paths.CASF_DOCKING_LIST)
    )
    logger.info("%s: scoring %d targets", baseline.name, len(targets))
    if allow:
        logger.info(
            "%s: %d target(s) excused as unscorable: %s",
            baseline.name, len(allow), ", ".join(allow),
        )

    paths.ensure_dirs()
    try:
        frame = baseline.score(targets)
    finally:
        baseline.teardown()

    out = args.results / "rescoring" / baseline.name / "pose_scores_measured.csv"
    write_pose_scores(frame, out)
    oriented = R.orient(frame, raw_col="head")
    logger.info(
        "%s: %d poses over %d targets -> %s",
        baseline.name,
        len(frame),
        frame["pdbid"].nunique(),
        out,
    )
    logger.info(
        "  DP@2A=%.1f DP@1A=%.1f rho=%.3f",
        R.docking_power(oriented, 2.0),
        R.docking_power(oriented, 1.0),
        R.ranking_rho(oriented),
    )


def _docking_targets(listing: Path) -> list[str]:
    """The docking-power target list, read without pulling in pandas."""
    targets = []
    for n, line in enumerate(listing.read_text().splitlines()):
        if not line.strip():
            continue
        if n == 0 and line.lstrip().startswith("#"):
            continue  # header
        targets.append(line.split()[0])
    return targets


if __name__ == "__main__":
    main()

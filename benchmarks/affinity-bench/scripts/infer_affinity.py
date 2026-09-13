"""Score the CASF-2016 core set with one affinity arm -> per-complex dumps.

Writes ``results/<arm>/<head-label>.csv``, one per head. Needs a GPU and the
source repo's checkpoints; run it under qsub.

Usage::

    .venv/bin/python scripts/infer_affinity.py --arm e250_kdki_mean --n-frames 16
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prolit.seeding import add_seed_argument, seed_from_args

from affinity_bench import inference
from affinity_bench.config import EvalConfig
from affinity_bench.io_dumps import write_affinity
from affinity_bench.variants import get

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="e250_kdki_mean")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--n-frames",
        type=int,
        default=1,
        help="Average the prediction over this many random rotations of the "
        "complex. 1 reproduces the published protocol.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Appended to the OUTPUT DIRECTORY name, so two protocols of one "
        "arm sit side by side as two rows (e.g. --n-frames 16 --suffix _f16). "
        "It names the directory rather than the file because every CSV inside "
        "one arm directory is treated as a member of that arm's ensemble.",
    )
    parser.add_argument(
        "--dataset",
        choices=("casf", "pdbbind-holdout"),
        default="casf",
        help="Evaluation set. 'pdbbind-holdout' scores the ~281 PDBbind Kd/Ki "
        "complexes that are in eval_holdout_pdbs.txt and so were excluded from "
        "our training corpora. The other ~2,074 are training data; scoring them "
        "would measure memorisation, which is why an id list is required rather "
        "than the directory being walked.",
    )
    parser.add_argument("--max-targets", type=int, default=None)
    add_seed_argument(parser)
    args = parser.parse_args()
    seed = seed_from_args(args)

    variant = get(args.arm)
    if not variant.heads:
        logger.error(
            "arm %s has no head checkpoints yet -- train one first", args.arm
        )
        return

    cfg = EvalConfig()
    cfg.affinity.n_frames = args.n_frames
    cfg.affinity.max_targets = args.max_targets

    complexes = None
    if args.dataset == "pdbbind-holdout":
        root = Path("/gs/bs/tga-ohuelab/ysato/data/P-L")
        complexes = inference.pdbbind_complexes(
            root,
            cfg.paths.source_repo / "data" / "pdbbind_holdout_eval.txt",
            tuple(root / n for n in ("test.csv", "val.csv")),
        )
        logger.info("pdbbind holdout: %d complexes", len(complexes))
        if not complexes:
            logger.error("no complexes resolved -- check the id list and root")
            return

    out_dir = args.results / f"{args.arm}{args.suffix}"
    for i, head in enumerate(variant.heads):
        df = inference.score_complexes(
            variant, cfg.paths, cfg.affinity, head_index=i, seed=seed,
            complexes=complexes,
        )
        path = out_dir / f"{head.label or f'head{i}'}.csv"
        write_affinity(df, path)
        logger.info("wrote %d complexes -> %s", len(df), path)


if __name__ == "__main__":
    main()

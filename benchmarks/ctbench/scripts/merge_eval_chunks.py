"""Merge chunked sbdd-bench eval outputs into one arm-level dump.

``scripts/jobs/eval_arm_chunk.sh`` splits an arm's 100 targets over N array
tasks, each writing its own ``results/generation/<arm>/chunk<K>/``. This
concatenates the per-molecule dumps into ``results/generation/<arm>/
per_molecule.parquet`` so :mod:`scripts.compare_arms` (which derives every
aggregate from the per-molecule rows) can read the arm as if it had been scored
in one job.

Usage::

    uv run python scripts/merge_eval_chunks.py joint_bo sep4096_bo
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RESULTS = Path("results/generation")


def merge(arm: str) -> None:
    chunks = sorted((RESULTS / arm).glob("chunk*/per_molecule.parquet"))
    if not chunks:
        print(f"[skip] {arm}: no chunk dumps under {RESULTS / arm}")
        return
    frames = [pd.read_parquet(c) for c in chunks]
    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["target_id", "idx"], keep="first")
    out = RESULTS / arm / "per_molecule.parquet"
    df.to_parquet(out, index=False)
    n_targets = df["target_id"].nunique()
    print(
        f"[merge] {arm}: {len(chunks)} chunks -> {len(df)} rows "
        f"({before - len(df)} dupes dropped), {n_targets} targets -> {out}"
    )


def main() -> None:
    arms = sys.argv[1:]
    if not arms:
        print(__doc__)
        raise SystemExit(2)
    for arm in arms:
        merge(arm)


if __name__ == "__main__":
    main()

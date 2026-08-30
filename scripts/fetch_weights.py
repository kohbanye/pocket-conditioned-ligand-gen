"""Download the published checkpoints and print the environment they belong in.

The point is that a fresh machine goes from ``uv sync`` to a running generation
without anyone reading a path out of a docstring::

    uv run python scripts/fetch_weights.py --group generate --env-file weights.env
    source weights.env

The exported ``SBDD_OWN_*`` variables are what an evaluation harness reads to
find the weights; this file names the variables but not the harness, because
scripts/ is the surface those harnesses drive and not the other way round.

The repo is private, so authenticate first (``hf auth login``, or set ``HF_TOKEN``).

Which file is which lives in :mod:`prolit.weights`, not here: pairing a
checkpoint with the wrong normalization statistics does not raise, it silently
produces coordinates at the wrong scale, so the pairing is the library's job and
this file only decides where the output goes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from prolit.weights import GROUPS, env_lines, fetch_group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        default="generate",
        choices=sorted(GROUPS),
        help="'generate' is enough to sample ligands; 'iterative' adds the MLM "
        "and its codebook-neighbour table; 'all' takes every refiner variant "
        "that appears in the comparison table.",
    )
    parser.add_argument("--repo", default=None, help="override the HF repo id")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="write the export lines here as well as printing them",
    )
    args = parser.parse_args()

    paths = fetch_group(args.group, repo=args.repo)
    width = max(len(n) for n in paths)
    for name, path in paths.items():
        print(f"{name:{width}s}  {path}")

    lines = env_lines(paths)
    if lines:
        print()
        for line in lines:
            print(line)
    if args.env_file is not None:
        args.env_file.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.env_file}")


if __name__ == "__main__":
    main()

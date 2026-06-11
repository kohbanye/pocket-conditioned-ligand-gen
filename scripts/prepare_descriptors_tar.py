"""Build the descriptor cache by streaming ligand tars (no extraction).

Low-inode alternative to ``prepare_descriptors.py``: reads molecules directly
from ``data/hub_cache/repo/ligands/*.tar`` instead of 25M extracted files.
Requires the snapshot (tars + manifest) and the extracted receptors.

Run::

    uv run python scripts/prepare_descriptors_tar.py \
        --repo-dir data/hub_cache/repo \
        --receptors-dir data/hub_cache/receptors \
        --cache-dir data/descriptor_cache_full \
        --source-types cdonly it0 it2_redocked other \
        --num-workers 40
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import PocketExtractionConfig
from src.data.tar_prep import prepare_descriptors_from_tars

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, default=Path("data/hub_cache/repo"))
    parser.add_argument(
        "--receptors-dir", type=Path, default=Path("data/hub_cache/receptors")
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--source-types", type=str, nargs="+", required=True)
    parser.add_argument("--num-workers", type=int, default=40)
    parser.add_argument(
        "--max-files-per-tar",
        type=int,
        default=None,
        help="Cap SDF files processed per tar (debug/smoke runs).",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir
    if (cache_dir / "shard_metadata.pt").exists():
        logger.info("Cache already exists at %s; nothing to do.", cache_dir)
        return
    cache_dir.mkdir(parents=True, exist_ok=True)

    total, counts = prepare_descriptors_from_tars(
        repo_dir=args.repo_dir,
        receptors_dir=args.receptors_dir,
        cache_dir=cache_dir,
        source_types=args.source_types,
        pocket_config=PocketExtractionConfig(),
        num_workers=args.num_workers,
        max_files_per_tar=args.max_files_per_tar,
    )
    logger.info("Cache written: %d complexes across %d shards", total, len(counts))


if __name__ == "__main__":
    main()

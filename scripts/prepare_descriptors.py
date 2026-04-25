"""Compute and shard descriptors into a user-chosen cache directory.

Thin wrapper around :meth:`ComplexDescriptorDataModule.prepare_data` that
lets the caller target a directory other than the default
``data/descriptor_cache``.  Used to rebuild the cache in parallel with an
in-flight training run (which is still reading the old cache) so that
downstream experiments can pick up a fresh cache without disruption.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import CrossDockedConfig, HubDatasetConfig, VQVAETrainingConfig
from src.data.descriptors import ComplexDescriptorDataModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Destination directory for the new cache (e.g. data/descriptor_cache_v2).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Multiprocessing pool size (overrides VQVAETrainingConfig.num_workers).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Limit to the first N protein-ligand pairs (sanity runs).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override CrossDockedConfig.data_dir (default: ./data).",
    )
    parser.add_argument(
        "--from-hub",
        action="store_true",
        help="Load source data via HubDatasetConfig (data/hub_cache/).",
    )
    parser.add_argument("--hub-repo-id", type=str, default=None)
    parser.add_argument(
        "--source-types",
        type=str,
        nargs="+",
        default=None,
        help="source_type filter (e.g. cdonly it0 it2_redocked).",
    )
    args = parser.parse_args()

    config = VQVAETrainingConfig()
    if args.num_workers is not None:
        config.num_workers = args.num_workers

    data_config = CrossDockedConfig()
    if args.max_pairs is not None:
        data_config.max_pairs = args.max_pairs
    if args.data_dir is not None:
        data_config.data_dir = args.data_dir

    hub_config = None
    if args.from_hub:
        hub_config = HubDatasetConfig()
        if args.hub_repo_id is not None:
            hub_config.repo_id = args.hub_repo_id
        if args.source_types is not None:
            hub_config.source_types = args.source_types

    dm = ComplexDescriptorDataModule(config, data_config, hub_config=hub_config)
    # Swap the cache_dir *after* construction so the DataModule writes the
    # new shards where the caller wants them.
    dm.cache_dir = args.cache_dir
    if dm.cache_dir.exists() and any(dm.cache_dir.iterdir()):
        logger.warning(
            "Cache dir %s is not empty; prepare_data will skip if shard metadata "
            "already exists there.",
            dm.cache_dir,
        )
    dm.cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Generating descriptor cache at %s (num_workers=%d, max_pairs=%s)",
        dm.cache_dir,
        config.num_workers,
        args.max_pairs,
    )
    dm.prepare_data()
    logger.info("Done. Cache written to %s", dm.cache_dir)


if __name__ == "__main__":
    main()

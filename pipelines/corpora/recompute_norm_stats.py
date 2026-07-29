"""Delete ``normalization_stats.pt`` and re-run the stats pass from shards.

Needed when the descriptor schema or ``continuous_mask`` changes without
regenerating the descriptor shards. The shards themselves store
pre-normalization values, so only the stats file — a tiny tensor dict —
is recomputed.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prolit.config import AtomVQVAETrainingConfig, CrossDockedConfig, HubDatasetConfig
from prolit.data.atom_descriptors import AtomComplexDescriptorDataModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override CrossDockedConfig.data_dir (default: ./data).",
    )
    parser.add_argument(
        "--from-hub",
        action="store_true",
        help="Use HubDatasetConfig so prepare_data paths match the usual training run.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override descriptor cache directory (default: data/descriptor_cache_allatom).",
    )
    args = parser.parse_args()

    config = AtomVQVAETrainingConfig()
    data_config = CrossDockedConfig()
    if args.data_dir is not None:
        data_config.data_dir = args.data_dir

    hub_config = HubDatasetConfig() if args.from_hub else None
    dm = AtomComplexDescriptorDataModule(config, data_config, hub_config=hub_config)
    if args.cache_dir is not None:
        dm.cache_dir = args.cache_dir

    stats_path = dm.cache_dir / "normalization_stats.pt"
    if stats_path.exists():
        stats_path.unlink()
        logger.info("Deleted stale stats file: %s", stats_path)
    else:
        logger.info("No existing stats file at %s (nothing to delete)", stats_path)

    dm.setup()
    logger.info("Recomputed stats: %s", stats_path)
    stats = dm.norm_stats
    if stats is None:
        msg = "Stats regeneration failed: dm.norm_stats is None after setup()"
        raise RuntimeError(msg)
    for name, tensor in stats.items():
        logger.info("  %-14s  %s", name, tensor.tolist())


if __name__ == "__main__":
    main()

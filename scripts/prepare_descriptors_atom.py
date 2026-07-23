"""Build the all-atom descriptor cache by streaming ligand tars (inode-safe).

All-atom + good-pose counterpart of ``scripts/prepare_descriptors_tar.py``:
expands the pocket to every heavy atom, derives Full ligand-parity chemistry
for protein atoms, and (by default) keeps only ``label == 1`` poses. Writes
``data/descriptor_cache_allatom`` consumed by ``scripts/train_vqvae_atom.py``.

Run (CPU, streams tars; needs the snapshot tars/manifest + extracted receptors)::

    uv run python scripts/prepare_descriptors_atom.py \
        --repo-dir data/hub_cache/repo \
        --receptors-dir data/hub_cache/receptors \
        --cache-dir data/descriptor_cache_allatom \
        --source-types cdonly --max-residues 50 --num-workers 40
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import PocketExtractionConfig
from src.data.atom_tar_prep import prepare_atom_descriptors_from_tars

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, default=Path("data/hub_cache/repo"))
    parser.add_argument(
        "--receptors-dir", type=Path, default=Path("data/hub_cache/receptors")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/descriptor_cache_allatom")
    )
    parser.add_argument("--source-types", type=str, nargs="+", default=["cdonly"])
    parser.add_argument("--num-workers", type=int, default=40)
    parser.add_argument(
        "--max-residues",
        type=int,
        default=50,
        help=(
            "Pocket residue cap. All-atom pockets are ~8x larger than backbone; "
            "50 keeps the per-complex sequence within the LM block_size while "
            "covering the observed 8 A pocket max (~55 residues)."
        ),
    )
    parser.add_argument("--distance-cutoff", type=float, default=8.0)
    parser.add_argument(
        "--include-decoys",
        action="store_true",
        help="Keep all poses (default keeps only label==1 good poses).",
    )
    parser.add_argument(
        "--keep-label1-docked",
        action="store_true",
        help=(
            "Also keep label==1 *_docked.sdf.gz files (all ~20 poses each). "
            "By default only *_min.sdf.gz minimized near-native poses are kept "
            "-- 'label' is per docking-run file, so a label==1 docked file "
            "still holds ~19 decoy poses."
        ),
    )
    parser.add_argument(
        "--max-files-per-tar",
        type=int,
        default=None,
        help="Cap SDF files processed per tar (debug/smoke runs).",
    )
    parser.add_argument(
        "--shard-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated tar shard indices to process (e.g. '0,1,2'). "
            "Restricts the run to those tars so a large full-pose build can be "
            "split across several jobs / calibrated on one tar. Default: all."
        ),
    )
    args = parser.parse_args()

    shard_ids = None
    if args.shard_ids is not None:
        shard_ids = [int(s) for s in args.shard_ids.split(",") if s.strip() != ""]

    cache_dir = args.cache_dir
    if (cache_dir / "shard_metadata.pt").exists():
        logger.info("Atom cache already exists at %s; nothing to do.", cache_dir)
        return
    cache_dir.mkdir(parents=True, exist_ok=True)

    pocket_config = PocketExtractionConfig(
        distance_cutoff=args.distance_cutoff,
        max_residues=args.max_residues,
    )
    total, counts = prepare_atom_descriptors_from_tars(
        repo_dir=args.repo_dir,
        receptors_dir=args.receptors_dir,
        cache_dir=cache_dir,
        source_types=args.source_types,
        pocket_config=pocket_config,
        num_workers=args.num_workers,
        good_poses_only=not args.include_decoys,
        min_only=not args.keep_label1_docked,
        max_files_per_tar=args.max_files_per_tar,
        shard_ids=shard_ids,
    )
    logger.info("Atom cache written: %d complexes across %d shards", total, len(counts))


if __name__ == "__main__":
    main()

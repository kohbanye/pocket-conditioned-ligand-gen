"""Training script for the unified all-atom VQ-VAE (one codebook).

Consumes the ``data/descriptor_cache_allatom`` shard cache built by
``scripts/prepare_descriptors_atom.py`` (run that first; this script does NOT
extract raw data, to stay inode-safe). Trains a single
:class:`~src.tokenizers.vqvae.TransformerVQVAE` (domain="atom") over protein +
ligand atoms.

Run (single GPU)::

    uv run python scripts/train_vqvae_atom.py \
        --source-types cdonly --cache-dir data/descriptor_cache_allatom \
        --codebook-size 8192 --mol-batch-size 256 --run-name atomvqvae-v1
"""

import argparse
import logging
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config import AtomVQVAETrainingConfig, CrossDockedConfig, HubDatasetConfig
from src.data.atom_descriptors import AtomComplexDescriptorDataModule
from src.model.vqvae_module import AtomVQVAEModule

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-types", type=str, nargs="+", default=["cdonly"])
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--codebook-size", type=int, default=None)
    parser.add_argument("--mol-batch-size", type=int, default=None)
    parser.add_argument(
        "--max-residues",
        type=int,
        default=None,
        help="Pocket residue cap (informational; the cache fixes the real one).",
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--include-decoys",
        action="store_true",
        help="Use all poses (default fold split is over the good-pose cache).",
    )
    args = parser.parse_args()

    config = AtomVQVAETrainingConfig()
    data_config = CrossDockedConfig()
    if args.codebook_size is not None:
        config.atom.codebook_size = args.codebook_size
    if args.mol_batch_size is not None:
        config.mol_batch_size = args.mol_batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.max_residues is not None:
        config.pocket.max_residues = args.max_residues
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs

    hub_config = HubDatasetConfig()
    hub_config.source_types = args.source_types
    hub_config.good_poses_only = not args.include_decoys

    torch.set_float32_matmul_precision("high")

    dm = AtomComplexDescriptorDataModule(config, data_config, hub_config=hub_config)
    if args.cache_dir is not None:
        dm.cache_dir = args.cache_dir
    if not (dm.cache_dir / "shard_metadata.pt").exists():
        msg = (
            f"Atom cache missing at {dm.cache_dir}. Run "
            "scripts/prepare_descriptors_atom.py first (inode-safe tar streaming)."
        )
        raise FileNotFoundError(msg)

    module = AtomVQVAEModule(config)
    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        precision=config.precision,
        logger=WandbLogger(project="pocket-ligand-vqvae", name=args.run_name),
        callbacks=[
            ModelCheckpoint(
                monitor="val/atom_coord",
                mode="min",
                save_top_k=3,
                filename="atomvqvae-{epoch:02d}-{val/atom_coord:.4f}",
            ),
        ],
    )
    trainer.fit(module, dm)
    trainer.test(module, dm)


if __name__ == "__main__":
    main()

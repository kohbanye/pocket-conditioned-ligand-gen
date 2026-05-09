"""Training script for joint protein + ligand VQ-VAE."""

import argparse
import logging
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config import CrossDockedConfig, HubDatasetConfig, VQVAETrainingConfig
from src.data.descriptors import ComplexDescriptorDataModule
from src.model.vqvae_module import VQVAEModule

logging.basicConfig(level=logging.INFO)


def main() -> None:  # noqa: C901, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--mol-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Training precision (bf16-mixed, 16-mixed, 32)",
    )
    parser.add_argument(
        "--from-hub",
        action="store_true",
        help="Load data from HuggingFace Hub",
    )
    parser.add_argument("--hub-repo-id", type=str, default=None, help="HF Hub repo ID")
    parser.add_argument(
        "--source-types",
        type=str,
        nargs="+",
        default=None,
        help="Types categories to use (e.g. cdonly it0)",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Wandb run name.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override descriptor cache directory.",
    )
    parser.add_argument(
        "--ligand-coord-weight",
        type=float,
        default=None,
        help="Override the ligand coord-MSE recon weight.",
    )
    parser.add_argument(
        "--protein-coord-weight",
        type=float,
        default=None,
        help="Override the protein coord-MSE recon weight.",
    )
    args = parser.parse_args()

    config = VQVAETrainingConfig()
    data_config = CrossDockedConfig()

    if args.max_pairs is not None:
        data_config.max_pairs = args.max_pairs
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.mol_batch_size is not None:
        config.mol_batch_size = args.mol_batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.precision is not None:
        config.precision = args.precision
    if args.ligand_coord_weight is not None:
        config.ligand.recon_weights["coord"] = args.ligand_coord_weight
    if args.protein_coord_weight is not None:
        config.protein.recon_weights["coord"] = args.protein_coord_weight

    hub_config = None
    if args.from_hub:
        hub_config = HubDatasetConfig()
        if args.hub_repo_id is not None:
            hub_config.repo_id = args.hub_repo_id
        if args.source_types is not None:
            hub_config.source_types = args.source_types

        from src.data import HubCrossDockedDataModule  # noqa: PLC0415

        hub_dm = HubCrossDockedDataModule(hub_config)
        hub_dm.prepare_data()

    # Enable TF32 for A100/H100 (free ~3x speedup for float32 matmuls).
    torch.set_float32_matmul_precision("high")

    dm = ComplexDescriptorDataModule(config, data_config, hub_config=hub_config)
    if args.cache_dir is not None:
        dm.cache_dir = args.cache_dir
    module = VQVAEModule(config)

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        precision=config.precision,
        logger=WandbLogger(project="pocket-ligand-vqvae", name=args.run_name),
        callbacks=[
            ModelCheckpoint(
                monitor="val/ligand_coord",
                mode="min",
                save_top_k=3,
                filename="vqvae-{epoch:02d}-{val/ligand_coord:.4f}",
            ),
        ],
    )
    trainer.fit(module, dm)
    trainer.test(module, dm)


if __name__ == "__main__":
    main()

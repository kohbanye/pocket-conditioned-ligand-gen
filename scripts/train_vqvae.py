"""Training script for joint protein + ligand VQ-VAE."""

import argparse
import logging

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config import CrossDockedConfig, HubDatasetConfig, VQVAETrainingConfig
from src.data.descriptors import ComplexDescriptorDataModule
from src.model.vqvae_module import VQVAEModule

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
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
    args = parser.parse_args()

    config = VQVAETrainingConfig()
    data_config = CrossDockedConfig()

    if args.max_pairs is not None:
        data_config.max_pairs = args.max_pairs
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers

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

    dm = ComplexDescriptorDataModule(config, data_config, hub_config=hub_config)
    module = VQVAEModule(config)

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        logger=WandbLogger(project="pocket-ligand-vqvae"),
        callbacks=[
            ModelCheckpoint(
                monitor="val/protein_recon",
                mode="min",
                save_top_k=3,
                filename="vqvae-{epoch:02d}-{val/protein_recon:.4f}",
            ),
        ],
    )
    trainer.fit(module, dm)
    trainer.test(module, dm)


if __name__ == "__main__":
    main()

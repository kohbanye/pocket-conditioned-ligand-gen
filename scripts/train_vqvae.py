"""Training script for joint protein + ligand VQ-VAE."""

import argparse
import logging

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config import CrossDockedConfig, VQVAETrainingConfig
from src.data.descriptors import ComplexDescriptorDataModule
from src.model.vqvae_module import VQVAEModule

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = VQVAETrainingConfig()
    data_config = CrossDockedConfig()

    if args.max_pairs is not None:
        data_config.max_pairs = args.max_pairs
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    dm = ComplexDescriptorDataModule(config, data_config)
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


if __name__ == "__main__":
    main()

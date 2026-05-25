"""Smoke test: load v4 cache and run a few training steps.

Verifies that ``ComplexDescriptorDataModule`` + multi-head ``VQVAEModule``
work end-to-end on a tiny shard. Intended as a debug / CI helper, not for
real training. Reports per-head losses after a handful of steps so the
recon weights can be sanity checked without spinning up a full TSUBAME run.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback

from src.config import CrossDockedConfig, HubDatasetConfig, VQVAETrainingConfig
from src.data.descriptors import ComplexDescriptorDataModule
from src.model.vqvae_module import VQVAEModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


class HeadLossPrinter(Callback):
    """Print per-head losses every N training steps and once at validation end."""

    def __init__(self, every_n_steps: int = 5) -> None:
        self.every_n_steps = every_n_steps

    def on_train_batch_end(  # noqa: PLR0913
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,  # noqa: ARG002
        outputs,  # noqa: ANN001, ARG002
        batch,  # noqa: ANN001, ARG002
        batch_idx: int,
    ) -> None:
        if batch_idx % self.every_n_steps != 0:
            return
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        keys = sorted(
            k for k in metrics if k.startswith(("train/protein_", "train/ligand_"))
        )
        head_keys = [k for k in keys if any(t in k for t in (
            "_coord", "_element", "_charge", "_hybrid", "_aromatic",
            "_ring", "_numH", "_aa", "_commit", "_total",
        ))]
        head_keys = [k for k in head_keys if "weight" not in k and "norm" not in k]
        msg = " ".join(f"{k.split('/')[-1]}={metrics[k]:.4f}" for k in head_keys[:14])
        logger.info("step=%d %s", trainer.global_step, msg)

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,  # noqa: ARG002
    ) -> None:
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        keys = sorted(k for k in metrics if k.startswith(("val/protein_", "val/ligand_")))
        head_keys = [k for k in keys if any(t in k for t in (
            "_coord", "_element", "_charge", "_hybrid", "_aromatic",
            "_ring", "_numH", "_aa", "_commit", "_total", "_codebook_util",
            "_perplexity",
        ))]
        if not head_keys:
            return
        msg = "  ".join(f"{k}={metrics[k]:.4f}" for k in head_keys)
        logger.info("== val == %s", msg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit-train-batches", type=int, default=10)
    parser.add_argument("--limit-val-batches", type=int, default=2)
    args = parser.parse_args()

    config = VQVAETrainingConfig()
    config.mol_batch_size = args.batch_size
    config.num_workers = 0
    config.max_epochs = args.max_epochs
    config.precision = "32"

    data_config = CrossDockedConfig()
    hub_config = HubDatasetConfig()  # only fold mapping; cache_dir override below
    dm = ComplexDescriptorDataModule(config, data_config, hub_config=hub_config)
    dm.cache_dir = args.cache_dir
    dm.setup()

    module = VQVAEModule(config)

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices=1,
        precision=config.precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=1,
        callbacks=[HeadLossPrinter(every_n_steps=2)],
    )

    torch.set_float32_matmul_precision("high")
    trainer.fit(module, dm)

    logger.info("=== Smoke test done ===")
    logger.info("If no exception was raised the v4 pipeline is wired up.")


if __name__ == "__main__":
    main()

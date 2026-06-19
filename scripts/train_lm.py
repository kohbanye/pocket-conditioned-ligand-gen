"""From-scratch training of the dense Qwen3 pocket-conditioned ligand LM.

Consumes the packed token cache from ``scripts/tokenize_dataset.py`` and trains
with next-token prediction (loss on all tokens). PyTorch Lightning + WandB, to
match the VQ-VAE training stack.

Run on TSUBAME node_f (4x H100, DDP)::

    uv run python scripts/train_lm.py --token-dir data/lm_tokens --run-name lm_v1

Two-stage curriculum: pretrain on GEOM, then fine-tune on CrossDocked by
warm-starting the LM weights (a fresh optimizer + LR schedule, *not* a resume)::

    # 1) pretrain (ligand-only GEOM tokens)
    uv run python scripts/train_lm.py --token-dir data/lm_tokens_geom \
        --run-name lm_pretrain --max-epochs 3
    # 2) fine-tune (CrossDocked complexes) from the pretrained weights
    uv run python scripts/train_lm.py --token-dir data/lm_tokens \
        --run-name lm_finetune --init-from <pretrain checkpoint>.ckpt
"""

from __future__ import annotations

import argparse
import logging

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config import LMTrainingConfig
from src.data.lm_dataset import LMTokenDataModule
from src.model.lm_module import LigandLMModule

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--accumulate", type=int, default=None)
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Warm-start LM weights from this checkpoint (e.g. a GEOM-pretrained "
        "run). Loads weights only -- a fresh optimizer + LR schedule, NOT a "
        "training resume. The model config (vocab/dims) must match.",
    )
    args = parser.parse_args()

    config = LMTrainingConfig()
    if args.token_dir is not None:
        config.token_dir = args.token_dir
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.micro_batch_size is not None:
        config.micro_batch_size = args.micro_batch_size
    if args.block_size is not None:
        config.block_size = args.block_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.accumulate is not None:
        config.gradient_accumulation = args.accumulate

    torch.set_float32_matmul_precision("high")

    dm = LMTokenDataModule(config)
    module = LigandLMModule(config)
    if args.init_from is not None:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        logging.getLogger(__name__).info(
            "Warm-started weights from %s (missing=%d, unexpected=%d)",
            args.init_from,
            len(missing),
            len(unexpected),
        )

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=config.precision,
        gradient_clip_val=config.grad_clip,
        accumulate_grad_batches=config.gradient_accumulation,
        logger=WandbLogger(project="pocket-ligand-lm", name=args.run_name),
        callbacks=[
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_top_k=3,
                # auto_insert_metric_name=False so the "/" in "val/loss" does not
                # create a nested checkpoint subdirectory.
                filename="lm-e{epoch:02d}-vl{val/loss:.4f}",
                auto_insert_metric_name=False,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
    )
    trainer.fit(module, dm)
    trainer.test(module, dm)


if __name__ == "__main__":
    main()

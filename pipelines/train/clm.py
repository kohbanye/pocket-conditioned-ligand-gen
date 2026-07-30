"""From-scratch training of the dense Qwen3 pocket-conditioned ligand LM.

Consumes the packed token cache from ``pipelines/corpora/`` and trains
with next-token prediction (loss on all tokens). PyTorch Lightning + WandB, to
match the VQ-VAE training stack.

Run on TSUBAME node_f (4x H100, DDP)::

    uv run python pipelines/train/clm.py --token-dir data/lm_tokens --run-name lm_v1

Two-stage curriculum: pretrain on GEOM, then fine-tune on CrossDocked by
warm-starting the LM weights (a fresh optimizer + LR schedule, *not* a resume)::

    # 1) pretrain (ligand-only GEOM tokens)
    uv run python pipelines/train/clm.py --token-dir data/lm_tokens_geom \
        --run-name lm_pretrain --max-epochs 3
    # 2) fine-tune (CrossDocked complexes) from the pretrained weights
    uv run python pipelines/train/clm.py --token-dir data/lm_tokens \
        --run-name lm_finetune --init-from <pretrain checkpoint>.ckpt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

from prolit.config import LMTrainingConfig
from prolit.data.lm_dataset import LMTokenDataModule
from prolit.model.lm_module import LigandLMModule
from prolit.seeding import add_seed_argument, seed_from_args

logging.basicConfig(level=logging.INFO)


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--accumulate", type=int, default=None)
    parser.add_argument(
        "--atom-codebook-size",
        type=int,
        default=None,
        help="Use the unified all-atom vocab (specials + this single codebook) "
        "instead of the legacy protein+ligand ranges. Match the token cache.",
    )
    parser.add_argument(
        "--mask-prompt",
        action="store_true",
        help="Condition-only fine-tuning: mask the <p> pocket prompt from the "
        "loss (loss only on the generated <l> block). Leave off for pretraining.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop when val/loss has not improved for this many checks (0=off). "
        "Use for fine-tuning so a held-out-pocket val picks a generalising model "
        "before it over-fits.",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Warm-start LM weights from this checkpoint (e.g. a GEOM-pretrained "
        "run). Loads weights only -- a fresh optimizer + LR schedule, NOT a "
        "training resume. The model config (vocab/dims) must match.",
    )
    add_seed_argument(parser)
    args = parser.parse_args()
    seed_from_args(args)

    config = LMTrainingConfig()

    # Recorded in the checkpoint's hparams, so a run remembers its seed.

    config.seed = args.seed
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
    if args.atom_codebook_size is not None:
        config.model.atom_codebook_size = args.atom_codebook_size
    config.mask_prompt = args.mask_prompt

    torch.set_float32_matmul_precision("high")

    # Pin the checkpoint dir to the run-name so downstream stages can warm-start
    # from a knowable path (not the W&B run id) for hold_jid chaining;
    # save_last gives a stable last.ckpt.
    ckpt_dir = (
        Path("pocket-ligand-lm") / args.run_name / "checkpoints"
        if args.run_name
        else None
    )
    callbacks: list = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            # auto_insert_metric_name=False so the "/" in "val/loss" does not
            # create a nested checkpoint subdirectory.
            filename="lm-e{epoch:02d}-vl{val/loss:.4f}",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if args.early_stop_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss", mode="min", patience=args.early_stop_patience
            )
        )

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
        deterministic=args.deterministic,
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=config.precision,
        gradient_clip_val=config.grad_clip,
        accumulate_grad_batches=config.gradient_accumulation,
        logger=WandbLogger(project="pocket-ligand-lm", name=args.run_name),
        callbacks=callbacks,
    )
    trainer.fit(module, dm)
    # Only run the final test pass if a test split exists (the mixed
    # pretraining cache is train/val only; the fine-tune cache has test).
    if (Path(config.token_dir) / "test.bin").exists():
        trainer.test(module, dm)


if __name__ == "__main__":
    main()

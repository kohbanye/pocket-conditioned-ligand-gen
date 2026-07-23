"""Fine-tune a pose-scoring head on the pretrained complex-token MLM encoder.

Warm-starts the ESM3-style encoder from an MLM checkpoint and trains an MLP head
to regress pose RMSD on rigid-perturbation decoys (:mod:`scripts.tokenize_decoys`).
The discriminative complement to zero-shot masked PLL.

Run (single GPU)::

    uv run python scripts/train_rescore.py \
        --token-dir data/lm_tokens_decoys \
        --mlm-ckpt pocket-ligand-mlm/j90rlrgm/checkpoints/mlm-e01-vl0.8528.ckpt \
        --run-name rescore_v1 --max-epochs 8
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

from src.config import ComplexMLMConfig, RescoreTrainingConfig
from src.data.rescore_dataset import RescoreDataModule
from src.model.rescore_module import ComplexRescoreModule

logging.basicConfig(level=logging.INFO)


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-dir", type=str, default=None)
    parser.add_argument(
        "--mlm-ckpt", type=str, default=None, help="Warm-start encoder."
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--atom-codebook-size", type=int, default=8192)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument(
        "--ranking-loss-weight",
        type=float,
        default=None,
        help="Add pairwise within-complex ranking loss (groups by RMSD==0 native).",
    )
    parser.add_argument("--complexes-per-batch", type=int, default=None)
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=None,
        help="Cap docs taken per group in a batch (affinity: ligands per protein).",
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "meanmax", "attn", "xattn", "pairsum"],
        default=None,
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Train only pooling+head (encoder fixed); pairs with ranking loss.",
    )
    parser.add_argument(
        "--efficiency",
        action="store_true",
        help="Regress pK / heavy-atom count (ligand efficiency) to strip the "
        "molecular-size confound; eval multiplies back by size.",
    )
    parser.add_argument(
        "--interaction-layers",
        type=int,
        default=None,
        help="Trainable transformer layers over the tokens before pooling.",
    )
    parser.add_argument(
        "--mlm-aux-weight",
        type=float,
        default=None,
        help="Weight of a masked-LM regularizer during affinity fine-tuning "
        "(lets a ranking loss adapt the encoder without collapsing it).",
    )
    parser.add_argument(
        "--label-cap",
        type=float,
        default=None,
        help="Clamp on the regression target. 8.0 suits RMSD decoys; raise to ~13 "
        "for pK affinity labels so real binders are not clipped.",
    )
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()

    config = RescoreTrainingConfig(
        model=ComplexMLMConfig(atom_codebook_size=args.atom_codebook_size)
    )
    if args.token_dir is not None:
        config.token_dir = args.token_dir
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.micro_batch_size is not None:
        config.micro_batch_size = args.micro_batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.ranking_loss_weight is not None:
        config.ranking_loss_weight = args.ranking_loss_weight
    if args.complexes_per_batch is not None:
        config.complexes_per_batch = args.complexes_per_batch
    if args.max_per_group is not None:
        config.max_per_group = args.max_per_group
    if args.freeze_encoder:
        config.freeze_encoder = True
    if args.efficiency:
        config.label_divide_by_size = True
    if args.interaction_layers is not None:
        config.head_interaction_layers = args.interaction_layers
    if args.mlm_aux_weight is not None:
        config.mlm_aux_weight = args.mlm_aux_weight
    if args.pooling is not None:
        config.pooling = args.pooling
    if args.label_cap is not None:
        config.rmsd_cap = args.label_cap

    torch.set_float32_matmul_precision("high")

    mlm_state = None
    if args.mlm_ckpt is not None:
        ckpt = torch.load(args.mlm_ckpt, map_location="cpu", weights_only=False)
        mlm_state = ckpt.get("state_dict", ckpt)

    # Pin the checkpoint dir to the run name. The default (a wandb run hash)
    # is non-deterministic, so a caller that has to *discover* which dir a job
    # produced races when several jobs run concurrently -- it once made a
    # meanmax job evaluate a sibling attn job's checkpoint. A run-name path is
    # unique per experiment and knowable in advance.
    ckpt_dir = (
        Path("pocket-ligand-rescore") / args.run_name / "checkpoints"
        if args.run_name
        else None
    )
    callbacks: list = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            filename="rescore-e{epoch:02d}-vl{val/loss:.4f}",
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

    dm = RescoreDataModule(config)
    module = ComplexRescoreModule(config, mlm_state=mlm_state)

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=config.precision,
        gradient_clip_val=config.grad_clip,
        accumulate_grad_batches=config.gradient_accumulation,
        logger=WandbLogger(project="pocket-ligand-rescore", name=args.run_name),
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(module, dm)


if __name__ == "__main__":
    main()

"""Train the E(3)-equivariant ligand pose refiner (flow-matching bridge).

Learns to map a VQ-VAE-corrupted ligand pose ``x0`` (the exact geometry the
generation pipeline emits) back onto the crystal native pose ``x1``, conditioned
on the frozen pocket -- a fast, differentiable, generation-time replacement for
Vina local minimisation. Data is produced by :mod:`scripts.tokenize_pose_refine`.

Run (single GPU)::

    uv run python scripts/train_pose_refine.py \
        --data-dir data/pose_refine \
        --run-name refine_v1 --max-epochs 40
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

from src.config import PoseRefinerConfig, PoseRefineTrainingConfig
from src.data.pose_refine_dataset import PoseRefineDataModule
from src.model.pose_refiner import PoseRefinerModule

logging.basicConfig(level=logging.INFO)


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--l-max", type=int, default=None)
    parser.add_argument("--pocket-cutoff", type=float, default=None)
    parser.add_argument("--lambda-clash", type=float, default=None)
    parser.add_argument("--lambda-pkt", type=float, default=None)
    parser.add_argument("--lambda-bond", type=float, default=None)
    parser.add_argument("--lambda-angle", type=float, default=None)
    parser.add_argument("--bridge-sigma", type=float, default=None)
    parser.add_argument("--online-jitter-sigma", type=float, default=None)
    parser.add_argument("--online-rigid-trans", type=float, default=None)
    parser.add_argument("--online-rigid-rot-deg", type=float, default=None)
    parser.add_argument("--online-rigid-prob", type=float, default=None)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()

    model = PoseRefinerConfig()
    if args.hidden_dim is not None:
        model.hidden_dim = args.hidden_dim
    if args.n_layers is not None:
        model.n_layers = args.n_layers
    if args.l_max is not None:
        model.l_max = args.l_max
    if args.pocket_cutoff is not None:
        model.pocket_cutoff = args.pocket_cutoff
    if args.lambda_clash is not None:
        model.lambda_clash = args.lambda_clash
    if args.lambda_pkt is not None:
        model.lambda_pkt = args.lambda_pkt
    if args.lambda_bond is not None:
        model.lambda_bond = args.lambda_bond
    if args.lambda_angle is not None:
        model.lambda_angle = args.lambda_angle
    if args.bridge_sigma is not None:
        model.bridge_sigma = args.bridge_sigma

    config = PoseRefineTrainingConfig(model=model)
    if args.data_dir is not None:
        config.data_dir = Path(args.data_dir)
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.micro_batch_size is not None:
        config.micro_batch_size = args.micro_batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.online_jitter_sigma is not None:
        config.online_jitter_sigma = args.online_jitter_sigma
    if args.online_rigid_trans is not None:
        config.online_rigid_trans = args.online_rigid_trans
    if args.online_rigid_rot_deg is not None:
        config.online_rigid_rot_deg = args.online_rigid_rot_deg
    if args.online_rigid_prob is not None:
        config.online_rigid_prob = args.online_rigid_prob
    if args.precision is not None:
        config.precision = args.precision

    torch.set_float32_matmul_precision("high")

    ckpt_dir = (
        Path("pocket-ligand-refine") / args.run_name / "checkpoints"
        if args.run_name
        else None
    )
    callbacks: list = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val/rmsd_refined",
            mode="min",
            save_top_k=3,
            filename="refine-e{epoch:02d}-r{val/rmsd_refined:.4f}",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if args.early_stop_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val/rmsd_refined",
                mode="min",
                patience=args.early_stop_patience,
            )
        )

    dm = PoseRefineDataModule(config)
    module = PoseRefinerModule(config)

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=config.precision,
        gradient_clip_val=config.grad_clip,
        accumulate_grad_batches=config.gradient_accumulation,
        logger=WandbLogger(project="pocket-ligand-refine", name=args.run_name),
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(module, dm)


if __name__ == "__main__":
    main()

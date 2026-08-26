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

from prolit.config import CLMTrainingConfig
from prolit.data.clm_dataset import CLMTokenDataModule
from prolit.model.clm_module import ProLITCLMModule
from prolit.provenance import RecordProvenance
from prolit.seeding import add_seed_argument, seed_from_args

logging.basicConfig(level=logging.INFO)


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--anchor-loss-weight",
        type=float,
        default=None,
        help="multiply the loss on the first --anchor-loss-atoms ligand tokens "
        "by this. The anchor is 1/25 of the loss and the model does not learn "
        "to read the pocket for it: its first-atom error with the correct "
        "pocket (2.13 A) equals that of a predictor that ignores the pocket "
        "and answers the mean ligand centroid (2.14 A).",
    )
    parser.add_argument("--anchor-loss-atoms", type=int, default=None)
    parser.add_argument(
        "--centroid-loss-weight",
        type=float,
        default=None,
        help="weight of an auxiliary head that regresses the ligand centroid "
        "from the <l> hidden state -- the state the first ligand token is "
        "predicted from. Up-weighting the anchor's own loss did nothing "
        "(p=0.40) because more gradient does not create a path from the pocket "
        "to that state; this asks for the quantity directly.",
    )
    parser.add_argument("--centroid-target", choices=("centroid", "anchor"),
                        default=None,
                        help="'anchor' regresses the FIRST ligand atom rather "
                        "than the molecule's centre. The head reaches 0.60 A on "
                        "the centroid, but the first atom is a median 1.98 A "
                        "away from it, so that accuracy does not transfer.")
    parser.add_argument("--code-mean-coords", type=str, default=None,
                        help="(n_codes, 3) table from code_mean_coords.py")
    parser.add_argument(
        "--code-geometry-tau",
        type=float,
        default=None,
        help="width (A) of a geometry-smoothed cross-entropy. Plain CE calls "
        "every wrong code equally wrong, but a code is a place: the LM's "
        "teacher-forced argmax lands atoms 2.52 A off the crystal where the "
        "quantizer alone lands them 0.35 A off, so what it gets wrong is "
        "which code is geometrically near. Needs --code-mean-coords.",
    )
    parser.add_argument(
        "--code-geometry-k",
        type=int,
        default=None,
        help="how many neighbours the smoothed target spreads over",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="stop after this many optimiser steps, validating and "
        "checkpointing every max_steps//4. One epoch of the full-pose corpus "
        "is 16.5M documents, so without this the smallest unit of evidence "
        "about a training change is a whole node-day.",
    )
    parser.add_argument(
        "--pocket-dropout",
        type=float,
        default=None,
        help="blank the pocket of this fraction of TRAINING documents, so the "
        "model also learns the unconditional distribution and generation can "
        "use classifier-free guidance. Measured on a model trained WITHOUT it, "
        "guidance only hurt (0.619 A at w=1 to 0.709 A at w=5) because the "
        "empty-pocket input was never seen.",
    )
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
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="GPUs to use (default auto = every GPU on the node). Unlike the "
        "VQ-VAE's IterableDataset, the packed token cache is a map-style "
        "Dataset, so Lightning injects a DistributedSampler itself and each "
        "rank gets a distinct 1/N shard -- the manual rank split that stream "
        "needed does not apply here.",
    )
    add_seed_argument(parser)
    args = parser.parse_args()
    seed_from_args(args)

    config = CLMTrainingConfig()

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
    if args.pocket_dropout is not None:
        config.pocket_dropout = args.pocket_dropout
    if args.anchor_loss_weight is not None:
        config.anchor_loss_weight = args.anchor_loss_weight
    if args.anchor_loss_atoms is not None:
        config.anchor_loss_atoms = args.anchor_loss_atoms
    if args.centroid_loss_weight is not None:
        config.centroid_loss_weight = args.centroid_loss_weight
    if args.code_mean_coords is not None:
        config.code_mean_coords = args.code_mean_coords
    if args.code_geometry_tau is not None:
        config.code_geometry_tau = args.code_geometry_tau
    if args.code_geometry_k is not None:
        config.code_geometry_k = args.code_geometry_k
    if args.centroid_target is not None:
        config.centroid_target = args.centroid_target
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
        # Writes run.json beside the checkpoints: command, git SHA, seed.
        RecordProvenance(seed=args.seed),
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

    dm = CLMTokenDataModule(config)
    module = ProLITCLMModule(config)
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
        **(
            {
                "max_steps": args.max_steps,
                "val_check_interval": max(1, args.max_steps // 4),
            }
            if args.max_steps
            else {}
        ),
        accelerator="auto",
        devices=args.devices if args.devices is not None else "auto",
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

"""From-scratch training of the bidirectional complex-token MLM (ESM-style).

Consumes the SAME packed token cache as ``pipelines/train/clm.py`` but serves one
complex per example and trains with a masked-token objective (loss on masked
positions only). This is the representation backbone for pose rescoring.

Run on TSUBAME (single GPU is usually enough for the ~110M default)::

    uv run python pipelines/train/mlm.py \
        --token-dir data/lm_tokens_finetune_mixed \
        --run-name mlm_allatom_v1 --max-epochs 10

Vocab mode must match the token cache: pass ``--atom-codebook-size 8192`` for
the single all-atom codebook caches (vocab 8199), or omit it for the legacy
protein+ligand 2-range caches (vocab 12295).
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

from prolit.config import MLMTrainingConfig
from prolit.data.mlm_dataset import MLMTokenDataModule
from prolit.model.mlm_module import ProLITMLMModule
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
    parser.add_argument("--accumulate", type=int, default=None)
    parser.add_argument("--mask-prob", type=float, default=None)
    parser.add_argument(
        "--mask-replace-prob",
        type=float,
        default=None,
        help="share of selected positions that become <mask>. Lower it (with "
        "--mask-random-prob raised) to train a REPAIRER rather than a filler: "
        "the sequence a generated pose arrives in has no <mask> in it, every "
        "token is present and merely wrong.",
    )
    parser.add_argument(
        "--mask-random-prob",
        type=float,
        default=None,
        help="share of selected positions replaced by another code. With "
        "--code-neighbours that code is a near miss, which is the error the "
        "CLM actually makes.",
    )
    parser.add_argument(
        "--code-neighbours",
        type=str,
        default=None,
        help="(n_codes, K) table of nearest codes in codebook space, used to "
        "draw the --mask-random-prob corruptions. Must come from the SAME "
        "VQ-VAE as the token corpus.",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--atom-codebook-size",
        type=int,
        default=None,
        help="Use the unified all-atom vocab (specials + this single codebook, "
        "e.g. 8192 -> vocab 8199) instead of the legacy protein+ligand ranges. "
        "Must match the token cache's meta.json.",
    )
    parser.add_argument(
        "--val-check-interval",
        type=float,
        default=1.0,
        help="Validate (and therefore checkpoint) this often within an epoch. "
        "An epoch of this corpus is ~7.5 h on one GPU, and Lightning only "
        "checkpoints at validation, so the default of 1.0 means a run that "
        "dies or hits its walltime mid-epoch leaves NOTHING -- which is exactly "
        "what happened twice on 2026-08-20 (once at 84%, once at 92%). Set it "
        "to 0.25 for long continuations so partial progress survives.",
    )
    parser.add_argument(
        "--mask-prob-max",
        type=float,
        default=None,
        help="upper end of a per-example uniform mask rate (lower end is "
        "--mask-prob). Unset keeps the fixed rate. 1.0 spans the schedule an "
        "iterative decoder walks through: a 15%%-only model decodes a fully "
        "masked ligand at RMSD 7.67 A against the causal model's 1.06.",
    )
    parser.add_argument(
        "--ligand-only-masking",
        action="store_true",
        help="Mask only ligand (<l>..</l>) tokens -> a condition-only MLM that "
        "models P(ligand | pocket). Leave off for full-complex pretraining.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop when val/loss has not improved for this many checks (0=off).",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Warm-start encoder weights from this checkpoint (weights only, "
        "fresh optimizer). The model config (vocab/dims) must match.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="stop after this many optimiser steps, and validate/checkpoint "
        "every max_steps//4. The corpus is 14.6M documents, so one epoch is "
        "the smallest unit this otherwise offers -- which makes it impossible "
        "to get a cheap read on whether a training change is working before "
        "committing a full node to it.",
    )
    parser.add_argument("--fast-dev-run", action="store_true")
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

    config = MLMTrainingConfig()

    # Recorded in the checkpoint's hparams, so a run remembers its seed.

    config.seed = args.seed
    if args.token_dir is not None:
        config.token_dir = Path(args.token_dir)
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
    if args.mask_prob is not None:
        config.mask_prob = args.mask_prob
    if args.mask_replace_prob is not None:
        config.mask_replace_prob = args.mask_replace_prob
    if args.mask_random_prob is not None:
        config.mask_random_prob = args.mask_random_prob
    if args.code_neighbours is not None:
        config.code_neighbours = args.code_neighbours
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.atom_codebook_size is not None:
        config.model.atom_codebook_size = args.atom_codebook_size
    config.ligand_only_masking = args.ligand_only_masking
    if args.mask_prob_max is not None:
        config.mask_prob_max = args.mask_prob_max

    torch.set_float32_matmul_precision("high")

    # Pin the checkpoint dir to the run-name so downstream head jobs can
    # reference this MLM by a knowable path (not the W&B run id) for hold_jid
    # chaining; save_last gives a stable last.ckpt.
    ckpt_dir = (
        Path("pocket-ligand-mlm") / args.run_name / "checkpoints"
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
            filename="mlm-e{epoch:02d}-vl{val/loss:.4f}",
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

    dm = MLMTokenDataModule(config)
    module = ProLITMLMModule(config)
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
        # --max-steps sets its own validation cadence (a quarter of the run), so
        # it wins; passing both would hand Trainer the same keyword twice.
        **(
            {
                "max_steps": args.max_steps,
                "val_check_interval": max(1, args.max_steps // 4),
            }
            if args.max_steps
            else {"val_check_interval": args.val_check_interval}
        ),
        accelerator="auto",
        devices=args.devices if args.devices is not None else "auto",
        precision=config.precision,
        gradient_clip_val=config.grad_clip,
        accumulate_grad_batches=config.gradient_accumulation,
        logger=WandbLogger(project="pocket-ligand-mlm", name=args.run_name),
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(module, dm)
    if not args.fast_dev_run and (Path(config.token_dir) / "test.bin").exists():
        trainer.test(module, dm)


if __name__ == "__main__":
    main()

"""Fine-tune a pose-scoring head on the pretrained complex-token MLM encoder.

Warm-starts the ESM3-style encoder from an MLM checkpoint and trains an MLP head
to regress pose RMSD on rigid-perturbation decoys
(:mod:`pipelines.corpora.tokenize_decoys`).
The discriminative complement to zero-shot masked PLL.

Run (single GPU)::

    uv run python pipelines/train/scoring_head.py \
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

from prolit.config import ProLITMLMConfig, RescoreTrainingConfig
from prolit.data.rescore_dataset import RescoreDataModule
from prolit.model.rescore_module import ComplexRescoreModule
from prolit.provenance import RecordProvenance
from prolit.seeding import add_seed_argument, seed_from_args

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
        "--listwise-weight",
        type=float,
        default=None,
        help="Weight of the within-complex ListNet loss (groups like --ranking).",
    )
    parser.add_argument("--listwise-tau", type=float, default=None)
    parser.add_argument(
        "--listwise-topk",
        type=int,
        default=None,
        help="Size of the EXTRA top-k listwise term (0 = off).",
    )
    parser.add_argument("--listwise-topk-weight", type=float, default=None)
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Train only the head (0.59M params) on top of a frozen, eval-mode "
        "encoder. Caps what can be fitted at what the pretrained representation "
        "exposes linearly, which is the point when the encoder memorizes a small "
        "corpus inside one epoch.",
    )
    parser.add_argument(
        "--listwise-higher-is-better",
        action="store_true",
        help="The label's GOOD end is its largest value (pK), not its smallest "
        "(RMSD). Sharpens the listwise term on strong binders instead of weak "
        "ones. Pass it for any affinity corpus.",
    )
    parser.add_argument("--listwise-topk-by-label", action="store_true")
    parser.add_argument("--listwise-topk-tau", type=float, default=None)
    parser.add_argument("--drop-native-pose", action="store_true")
    parser.add_argument("--complexes-per-batch", type=int, default=None)
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=None,
        help="Cap docs taken per group in a batch (affinity: ligands per protein).",
    )
    parser.add_argument(
        "--efficiency",
        action="store_true",
        help="Regress pK / heavy-atom count (ligand efficiency) to strip the "
        "molecular-size confound; eval multiplies back by size.",
    )
    parser.add_argument(
        "--label-cap",
        type=float,
        default=None,
        help="Clamp on the regression target. 8.0 suits RMSD decoys; raise to ~13 "
        "for pK affinity labels so real binders are not clipped.",
    )
    parser.add_argument(
        "--max-label",
        type=float,
        default=None,
        help="Train only on poses with label <= this (near-native specialist).",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Checkpoint to resume from (restores weights, optimizer and epoch), "
        "so a run interrupted mid-training continues instead of restarting.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Max tokens per doc. Must exceed the corpus max_len or long "
        "complexes (big ligands) get their ligand tail truncated away.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Train on at most this many docs (corpus-size ablation).",
    )
    parser.add_argument("--fast-dev-run", action="store_true")
    add_seed_argument(parser)
    parser.add_argument("--encoder-lr-scale", type=float, default=1.0)
    parser.add_argument(
        "--group-mean-weight",
        type=float,
        default=0.0,
        help="Weight of a smooth-L1 term on each protein group's MEAN prediction "
        "(0 = off). Targets the measured deficit: target-level differences come "
        "out ~19%% too small, and per-complex regression does not penalise a "
        "uniform per-target offset. Switching this on also switches batching to "
        "whole groups, so a control must use the same batching with weight 0.",
    )
    parser.add_argument(
        "--corr-weight",
        type=float,
        default=0.0,
        help="Batch-wise Pearson term: loss = (1-w)*smooth_l1 + w*(1-PCC). 0 = off. "
        "OnionNet-2 trains with w=0.7 by default and GenScore carries a corrcoef "
        "term; a pure per-complex regression never asks predictions to line up on "
        "a common scale, and squared error is minimised by shrinking toward the "
        "mean when the signal is uncertain.",
    )
    parser.add_argument(
        "--group-batch-partition",
        action="store_true",
        help="With group batching on, split a group larger than --max-per-group "
        "across batches instead of drawing one random slice per epoch. Keeps the "
        "epoch covering every document once; without it the union corpus with "
        "real protein groups shows 46%% of its documents per epoch.",
    )
    parser.add_argument(
        "--rotations-per-complex",
        type=int,
        default=1,
        help="Orientations per complex held by the TRAIN corpus (built with "
        "--num-rotations). >1 indexes one entry per complex and draws one "
        "orientation per access, so an epoch stays complex-sized and the frame "
        "varies across epochs -- unlike training on R times the documents, "
        "which memorised the corpus. Val always uses one orientation.",
    )
    parser.add_argument(
        "--readout",
        default="ligand",
        help="Which encoder states the head averages: 'ligand' (every existing "
        "checkpoint), 'balanced' (pocket mean and ligand mean, equally "
        "weighted), 'allatoms', or 'concat' (pocket mean and ligand mean kept side "
        "by side; widens the head's first layer, so loading it under another "
        "readout fails on shape). The first three are parameter-free. The pooling a "
        "head is TRAINED under has to be the pooling it is scored under, so "
        "this value belongs on the arm that names the checkpoint.",
    )
    parser.add_argument(
        "--layer-mix",
        action="store_true",
        help="Let the head learn a softmax mix over the embedding output and "
        "every encoder layer instead of reading the last layer only. The last "
        "layer is the one the MLM objective specialised; intermediate layers "
        "are often the better transfer features. Adds one weight per state "
        "(14 for a 13-layer encoder) and nothing else, and the encoder stays "
        "frozen. A checkpoint trained with this flag must be scored with it.",
    )
    parser.add_argument(
        "--layer-mix-init-last",
        type=float,
        default=0.0,
        help="With --layer-mix: starting logit of the LAST state, the others "
        "starting at 0. 4.0 puts ~0.81 of the mix on the last layer, so "
        "training starts roughly where a last-layer head already is. 0 (the "
        "default) starts at the uniform average, which measured worse.",
    )
    parser.add_argument(
        "--layer-mix-lr-scale",
        type=float,
        default=1.0,
        help="With --layer-mix: multiply the learning rate of the mixing "
        "scalars (and nothing else) by this. At 1.0 they did not move at all.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=0,
        help="Adapt the frozen encoder with rank-r LoRA on every attention "
        "block's QKV and output projections (0 disables). The encoder's own "
        "weights never change and the adapters start as a no-op.",
    )
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument(
        "--distill-path",
        default="",
        help="float32 .npy of teacher predictions, one per TRAINING document "
        "in file order (experiments/.../dump_teacher_preds.py).",
    )
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument(
        "--input-mask-prob",
        type=float,
        default=0.0,
        help="Replace this share of the TRAIN split's codebook tokens with the "
        "encoder's <mask> id (0 disables; val and inference never mask). "
        "Unlike a crop or chemistry change this stays inside the pretraining "
        "distribution -- <mask> is what the MLM saw at 15%% of positions -- so "
        "0.15 is the rate that matches it.",
    )
    parser.add_argument(
        "--input-mask-side",
        default="ligand",
        help="Which side --input-mask-prob may hide: 'ligand' (the rows the "
        "default readout pools, so the encoder has to fill them from the "
        "pocket), 'pocket' (the side carrying the between-target scale), or "
        "'both'.",
    )
    parser.add_argument(
        "--input-mask-pool",
        default="keep",
        help="With --input-mask-prob: 'keep' leaves a masked LIGAND position in "
        "the pooled mean, so the readout averages the state the encoder "
        "inferred there; 'drop' removes it, making this plain input dropout "
        "whose pooled statistic matches inference. Masked POCKET rows are "
        "dropped either way -- they stop being codebook ids.",
    )
    args = parser.parse_args()
    seed_from_args(args)

    config = RescoreTrainingConfig(
        model=ProLITMLMConfig(atom_codebook_size=args.atom_codebook_size)
    )
    # Recorded in the checkpoint's hparams, so a run remembers its seed.
    config.seed = args.seed
    if args.token_dir is not None:
        config.token_dir = args.token_dir
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.micro_batch_size is not None:
        config.micro_batch_size = args.micro_batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    config.encoder_lr_scale = args.encoder_lr_scale
    config.rotations_per_complex = args.rotations_per_complex
    config.group_mean_weight = args.group_mean_weight
    config.corr_weight = args.corr_weight
    config.layer_mix_init_last = args.layer_mix_init_last
    config.layer_mix_lr_scale = args.layer_mix_lr_scale
    config.lora_rank = args.lora_rank
    config.lora_alpha = args.lora_alpha
    config.distill_path = args.distill_path
    config.distill_weight = args.distill_weight
    config.group_batch_partition = args.group_batch_partition
    config.input_mask_prob = args.input_mask_prob
    config.input_mask_side = args.input_mask_side
    config.input_mask_pool = args.input_mask_pool
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.listwise_weight is not None:
        config.listwise_loss_weight = args.listwise_weight
    if args.listwise_topk is not None:
        config.listwise_topk = args.listwise_topk
    if args.listwise_topk_weight is not None:
        config.listwise_topk_weight = args.listwise_topk_weight
    if args.freeze_encoder:
        config.freeze_encoder = True
    if args.listwise_higher_is_better:
        config.listwise_higher_is_better = True
    if args.listwise_topk_by_label:
        config.listwise_topk_by_label = True
    if args.listwise_topk_tau is not None:
        config.listwise_topk_tau = args.listwise_topk_tau
    if args.drop_native_pose:
        config.drop_native_pose = True
    if args.listwise_tau is not None:
        config.listwise_label_tau = args.listwise_tau
        config.listwise_pred_tau = args.listwise_tau
    if args.complexes_per_batch is not None:
        config.complexes_per_batch = args.complexes_per_batch
    if args.max_per_group is not None:
        config.max_per_group = args.max_per_group
    if args.efficiency:
        config.label_divide_by_size = True
    if args.block_size is not None:
        config.block_size = args.block_size
    if args.max_docs is not None:
        config.max_docs = args.max_docs
    if args.max_label is not None:
        config.max_label = args.max_label
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
    # sweep point evaluate a sibling point's checkpoint. A run-name path is
    # unique per experiment and knowable in advance.
    ckpt_dir = (
        Path("pocket-ligand-rescore") / args.run_name / "checkpoints"
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
    module = ComplexRescoreModule(
        config,
        mlm_state=mlm_state,
        readout=args.readout,
        layer_mix=args.layer_mix,
        lora_rank=args.lora_rank,
    )

    trainer = L.Trainer(
        deterministic=args.deterministic,
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
    trainer.fit(module, dm, ckpt_path=args.resume_from)


if __name__ == "__main__":
    main()

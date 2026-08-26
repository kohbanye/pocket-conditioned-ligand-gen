"""Training script for the unified all-atom VQ-VAE (one codebook).

Consumes the ``data/descriptor_cache_allatom`` shard cache built by
``pipelines/corpora/build_descriptors.py`` (run that first; this script does NOT
extract raw data, to stay inode-safe). Trains a single
:class:`~prolit.tokenizers.vqvae.TransformerVQVAE` (domain="atom") over protein +
ligand atoms.

Run (single GPU)::

    uv run python pipelines/train/vqvae.py \
        --source-types cdonly --cache-dir data/descriptor_cache_allatom \
        --codebook-size 8192 --mol-batch-size 256 --run-name atomvqvae-v1
"""

import argparse
import logging
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from prolit.config import AtomVQVAETrainingConfig, CrossDockedConfig, HubDatasetConfig
from prolit.data.atom_descriptors import AtomComplexDescriptorDataModule
from prolit.model.vqvae_module import AtomVQVAEModule
from prolit.provenance import RecordProvenance
from prolit.seeding import add_seed_argument, seed_from_args

logging.basicConfig(level=logging.INFO)


def _resume_path(resume_from: Path | None) -> str | None:
    """Validate ``--resume-from`` and return it as a ``trainer.fit`` ckpt_path.

    Failing loudly matters here: a typo'd path silently starting from scratch
    would burn the whole job before anyone noticed.
    """
    if resume_from is None:
        return None
    if not resume_from.exists():
        msg = f"--resume-from checkpoint missing: {resume_from}"
        raise SystemExit(msg)
    return str(resume_from)


def _apply_weight_overrides(
    config: AtomVQVAETrainingConfig, overrides: list[str]
) -> None:
    """Apply ``--weight KEY=VALUE`` onto ``recon_weights``, refusing typos.

    A misspelled key would otherwise train a full run at the defaults and read
    as a null result for the point on the Pareto front it was meant to test.
    """
    for override in overrides:
        key, _, value = override.partition("=")
        if key not in config.atom.recon_weights:
            msg = (
                f"--weight {override!r}: {key!r} is not a recon weight. "
                f"Known: {sorted(config.atom.recon_weights)}"
            )
            raise SystemExit(msg)
        config.atom.recon_weights[key] = float(value)


def build_parser() -> argparse.ArgumentParser:
    """Every knob of a VQ-VAE run, so ``main`` only wires them up."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-types", type=str, nargs="+", default=["cdonly"])
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "checkpoint to resume training from (typically the run's own "
            "last.ckpt). Restores optimizer, LR-scheduler and epoch counter, so "
            "training continues rather than restarting -- passed to "
            "trainer.fit(ckpt_path=...), not load_from_checkpoint."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="load weights from this checkpoint and start a fresh run. Unlike "
        "--resume-from this carries no optimizer or epoch state, which is what "
        "a decoder-only fine-tune of a finished tokenizer needs.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="train the decoder only, holding the encoder and codebook where "
        "the loaded checkpoint left them. The codes do not move, so every "
        "token stream and language model built on them stays valid.",
    )
    parser.add_argument(
        "--balanced-chem-loss",
        action="store_true",
        help="weight each categorical head's atoms by the inverse frequency of "
        "their own class, so a head cannot satisfy its loss by always "
        "answering with the majority",
    )
    parser.add_argument(
        "--codebook-size",
        type=int,
        default=None,
        help="Protein codebook size (also the sole codebook when not --split).",
    )
    parser.add_argument("--mol-batch-size", type=int, default=None)
    parser.add_argument(
        "--max-residues",
        type=int,
        default=None,
        help="Pocket residue cap (informational; the cache fixes the real one).",
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="GPUs to use (default auto = all). Measured on 4 epochs, batch "
        "256 per device: node_f 1 GPU 23.1 min (0.385 pt), 2 GPU 13.8 min "
        "(0.230), 4 GPU 9.1 min (0.152), and gpu_1 1 GPU 23.5 min (0.098). "
        "Scaling is sublinear -- 1.67x on two devices, 2.54x on four -- because "
        "every rank still opens every shard file, so only the compute divides. "
        "Pick 4 devices on node_f for turnaround, gpu_1 for cost; they differ "
        "by 1.5x in billing and 2.5x in wall clock.",
    )
    parser.add_argument(
        "--include-decoys",
        action="store_true",
        help="Use all poses (default fold split is over the good-pose cache).",
    )
    parser.add_argument(
        "--predict-knn-offsets",
        action="store_true",
        help="Reconstruct the K nearest-neighbour displacements as well as the "
        "pocket-anchored position, so local geometry is supervised (R1). Adds "
        "one head, so the weights are not interchangeable with a run without it.",
    )
    parser.add_argument(
        "--bond-distance-loss",
        action="store_true",
        help="Penalise the decoded 1-2 and 1-3 distances against the reference "
        "(R1, corrected). Constrains the coord head itself, unlike "
        "--predict-knn-offsets which adds a parallel output and was measured to "
        "change bond accuracy not at all.",
    )
    parser.add_argument(
        "--bond-distance-all-sources",
        action="store_true",
        help="Apply the 1-2/1-3 distance terms to protein atoms too, not just "
        "ligand ones. lDDT is a local-distance score, and the ligand-only term "
        "measurably cost the protein (TM 0.826 -> 0.809). The clash floor stays "
        "ligand-only. Requires --bond-distance-loss.",
    )
    parser.add_argument(
        "--local-distance-loss",
        action="store_true",
        help="Score the union of 1-2 and 1-3 pairs with one relative-error "
        "term, dropping the 5.0/2.0 ratio between --bond-distance-loss's two. "
        "Applies to every atom, so no per-source split.",
    )
    parser.add_argument(
        "--local-ligand-only",
        action="store_true",
        help="Keep the local term on ligand rows, as bond12/bond13 were. Over "
        "every atom it left the protein untouched but took ligand PB-valid "
        "from 0.685 to 0.124: at 9.3 protein atoms per ligand atom, the "
        "ligand's bonds are outvoted inside a single term.",
    )
    parser.add_argument(
        "--drop-clash",
        action="store_true",
        help="Remove the clash hinge, which carries the last hand-set weight "
        "(5.0) in the geometry objective. Worth testing once 1-2/1-3 are "
        "constrained, rather than assuming either way.",
    )
    parser.add_argument(
        "--pair-distance-loss",
        action="store_true",
        help="One relative-error term over every pair within 15 A, all atoms, "
        "in place of --bond-distance-loss / --distance-map-loss / the clash "
        "hinge. Measuring error relative to the reference distance weights "
        "proximity by the form of the expression, so the 5.0/2.0/5.0 hand "
        "weights and the per-source split both go away.",
    )
    parser.add_argument(
        "--keep-clash",
        action="store_true",
        help="Keep the clash hinge alongside --pair-distance-loss, to test "
        "whether the pairwise term really subsumes it.",
    )
    parser.add_argument(
        "--distance-map-loss",
        action="store_true",
        help="Penalise every protein-protein distance the reference keeps under "
        "15 A (lDDT's own inclusion radius). The bonded terms call a C-C bond at "
        "1.92 A while backbone lDDT scores CA-CA pairs at ~3.8 A, so they never "
        "touch what that metric measures; this does.",
    )
    parser.add_argument(
        "--loss-balancing",
        choices=["none", "scale", "constrained", "uncertainty"],
        default="none",
        help="How the per-head losses are combined. 'scale' divides each by a "
        "running mean of itself so no head's raw magnitude decides its "
        "influence -- the heads span four orders, which is why hand weights "
        "existed. 'constrained' instead treats geometry as the objective and "
        "holds each chemistry head at the level vq_ctrl_p3 reached. "
        "'uncertainty' (Kendall et al.) was measured to diverge here.",
    )
    parser.add_argument(
        "--pocket-order",
        choices=("sequence", "distance"),
        default="sequence",
        help="order of pocket residues inside the descriptor cache. "
        "'distance' puts the residues nearest the ligand LAST. Changes the "
        "cache, so a cache built one way must not be reused with the other.",
    )
    parser.add_argument(
        "--pocket-context",
        action="store_true",
        help="widen each ligand atom's knn SEARCH set to the pocket, filling "
        "its empty neighbour slots with the nearest protein atoms. Descriptor "
        "stays 33-D. THE CACHE MUST MATCH: a cache built without this flag "
        "holds ligand-only neighbours and cannot be reused with it.",
    )
    parser.add_argument(
        "--clash-floor",
        type=float,
        default=None,
        help="Minimum distance (A) enforced between ligand atom pairs that the "
        "reference keeps at least that far apart. Default 1.2 is the historical "
        "value and is nearly inert; 2.4 targets the invented contacts.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help="Override one entry of recon_weights, repeatable "
        "(--weight bond12=20.6 --weight dmap=2.4). The hand-set geometry "
        "weights were searched with pipelines/train/tune_vqvae.py, whose "
        "Pareto front is a straight protein/ligand trade-off, so a run has to "
        "be able to say which point on it it sits at. Unknown keys are "
        "refused: a typo would otherwise train a full run at the defaults and "
        "look like a null result.",
    )
    parser.add_argument(
        "--ligand-source-weight",
        type=float,
        default=None,
        help="Weight of ligand atoms in the reconstruction loss (R2). "
        "CrossDocked supplies 8.3 protein atoms per ligand atom, so 8.3 "
        "equalises the two; 1.0 reproduces the published runs.",
    )
    parser.add_argument(
        "--ligand-ema-weight",
        type=float,
        default=None,
        help="Weight of ligand atoms in the codebook EMA alone. Defaults to "
        "--ligand-source-weight, which is how the two were coupled. Set both "
        "separately to tell apart 'the ligand has too few codes' from 'the "
        "encoder is not pushed to use them'.",
    )
    parser.add_argument(
        "--exclude-eval-pdbs",
        action="store_true",
        help="Hold out every complex whose receptor PDB id appears in CASF-2016 "
        "or the sbdd-bench targets. The CrossDocked fold's own test side is "
        "excluded regardless. Published runs were trained WITHOUT this, and 169 "
        "of the 285 CASF entries sit on the fold-0 train side, so they saw them.",
    )
    parser.add_argument(
        "--modality",
        choices=["both", "protein", "ligand"],
        default="both",
        help="Ablation: train on both atom streams (joint, default) or a single "
        "modality (protein-only / ligand-only) on the SAME complexes. "
        "Single-modality runs write their own normalization_stats_<modality>.pt.",
    )
    add_seed_argument(parser)
    return parser


def main() -> None:  # noqa: C901, PLR0915
    args = build_parser().parse_args()
    seed_from_args(args)

    config = AtomVQVAETrainingConfig()

    # Recorded in the checkpoint's hparams, so a run remembers its seed.

    config.seed = args.seed
    data_config = CrossDockedConfig()
    if args.codebook_size is not None:
        config.atom.codebook_size = args.codebook_size
    if args.mol_batch_size is not None:
        config.mol_batch_size = args.mol_batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.max_residues is not None:
        config.pocket.max_residues = args.max_residues
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    config.atom.predict_knn_offsets = args.predict_knn_offsets
    config.atom.bond_distance_loss = args.bond_distance_loss
    config.atom.bond_distance_all_sources = args.bond_distance_all_sources
    config.atom.distance_map_loss = args.distance_map_loss
    config.atom.local_distance_loss = args.local_distance_loss
    config.atom.drop_clash = args.drop_clash
    config.atom.local_distance_ligand_only = args.local_ligand_only
    config.atom.pair_distance_loss = args.pair_distance_loss
    config.atom.keep_clash = args.keep_clash
    config.atom.loss_balancing = args.loss_balancing
    if args.clash_floor is not None:
        config.atom.clash_floor = args.clash_floor
    config.pocket.pocket_context = args.pocket_context
    config.pocket.pocket_order = args.pocket_order
    _apply_weight_overrides(config, args.weight)
    if args.ligand_source_weight is not None:
        config.atom.ligand_source_weight = args.ligand_source_weight
    config.atom.ligand_ema_weight = args.ligand_ema_weight

    hub_config = HubDatasetConfig()
    hub_config.source_types = args.source_types
    hub_config.good_poses_only = not args.include_decoys
    hub_config.exclude_eval_pdbs = args.exclude_eval_pdbs

    torch.set_float32_matmul_precision("high")

    dm = AtomComplexDescriptorDataModule(
        config, data_config, hub_config=hub_config, modality=args.modality
    )
    if args.cache_dir is not None:
        dm.cache_dir = args.cache_dir
    if not (dm.cache_dir / "shard_metadata.pt").exists():
        msg = (
            f"Atom cache missing at {dm.cache_dir}. Run "
            "pipelines/corpora/build_descriptors.py first (inode-safe tar streaming)."
        )
        raise FileNotFoundError(msg)

    config.atom.freeze_encoder = args.freeze_encoder
    config.atom.balanced_chem_loss = args.balanced_chem_loss
    module = AtomVQVAEModule(config)
    if args.init_from is not None:
        state = torch.load(args.init_from, map_location="cpu", weights_only=False)
        missing, unexpected = module.load_state_dict(
            state["state_dict"], strict=False
        )
        # Loud rather than strict: a decoder-only run deliberately changes the
        # config (balanced loss, frozen encoder) and those do not add weights,
        # so anything reported here is a real mismatch worth seeing in the log.
        print(f"[init] from {args.init_from}")
        print(f"[init] missing={len(missing)} unexpected={len(unexpected)}")
        for k in list(missing)[:5] + list(unexpected)[:5]:
            print(f"[init]   {k}")
    # Pin the checkpoint dir to the run-name so downstream jobs can find it
    # without knowing the auto-generated W&B run id (needed for the ablation
    # pipeline chaining). save_last gives a fixed last.ckpt path too.
    ckpt_dir = (
        Path("pocket-ligand-vqvae") / args.run_name / "checkpoints"
        if args.run_name
        else None
    )
    # A single-modality run leaves whole heads unused, and DDP refuses to run
    # with parameters that produced no gradient. ``aa`` and ``bb_sc`` are scored
    # on protein rows only, so a ligand-only run never touches them and the
    # reducer aborts on the first step. The joint run has no such head and keeps
    # the faster default; the protein-only run happens not to trip it either,
    # because what IT switches off (1-2/1-3, clash) are ligand-row terms read
    # off the shared coord output rather than heads of their own.
    # ``--devices`` unset means "however many this node has", which on node_f is
    # four, so an unset value has to count as multi-GPU here rather than as one.
    strategy = (
        "ddp_find_unused_parameters_true"
        if args.modality != "both" and args.devices != 1
        else "auto"
    )
    trainer = L.Trainer(
        deterministic=args.deterministic,
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices=args.devices if args.devices is not None else "auto",
        strategy=strategy,
        precision=config.precision,
        logger=WandbLogger(project="pocket-ligand-vqvae", name=args.run_name),
        callbacks=[
            # Writes run.json beside the checkpoints: command, git SHA, seed.
            RecordProvenance(seed=args.seed),
            ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor="val/atom_coord",
                mode="min",
                save_top_k=3,
                save_last=True,
                filename="atomvqvae-{epoch:02d}-{val/atom_coord:.4f}",
            ),
        ],
    )
    trainer.fit(module, dm, ckpt_path=_resume_path(args.resume_from))
    trainer.test(module, dm)


if __name__ == "__main__":
    main()

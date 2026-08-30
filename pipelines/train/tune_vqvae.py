"""Search the VQ-VAE's geometry weights instead of setting them by hand.

``recon_weights`` carries five numbers nobody derived -- coord 1.0, bond12 5.0,
bond13 2.0, clash 5.0, dmap 1.0 -- and every attempt to justify them from the
data has failed differently. Normalising each term by its variance implies a
bond12:dmap ratio of 1200:1, which deletes the distance map that the protein
side depends on; normalising by the evaluation's tolerances lands within 4x of
the hand values but is an argument, not a measurement. So measure them.

**The objectives are the two unweighted diagnostics, never the training loss.**
Changing the weights changes what ``val_total`` means, so it cannot rank trials.
``dmap_mae`` (protein, every pair inside 15 A) and ``bond12_mae`` (ligand, 1-2
distances) are plain mean absolute errors in Angstroms: they mean the same thing
whatever the weights were. They are also the two the benchmark actually cares
about -- lDDT is a local-distance metric and bond perception decides whether a
SMILES comes back at all.

Two objectives, not one scalarisation. Every attempt so far to trade the protein
against the ligand has produced a frontier rather than a winner (3.0 ligand
weight beats 8.3 on PoseBusters and loses on SMILES), so a Pareto front is the
honest output and the choice of point on it belongs to the paper, not here.

**The short proxy is a screen, not a verdict.** Ranking at a few dozen epochs
has already been measured to disagree with ranking at 250: at 100 epochs the
separate tokenizer beat joint on PoseBusters (0.279 vs 0.128) and SMILES (0.309
vs 0.104), and at 250 both reversed. Treat the front as a shortlist and confirm
the pick at full length before it becomes a paper number.

Workers share one study over a journal file, so N jobs against the same
``--study-dir`` cooperate::

    python jobs/submit.py --name tune --resource gpu_1 --hours 6 \\
        --sweep worker=0,1,2,3 -- \\
        pipelines/train/tune_vqvae.py --study-dir pocket-ligand-vqvae/tune_geom \\
        --trial-epochs 20 --seed 7

SQLite is deliberately not used: the study lives on Lustre, where its locking is
unreliable. ``JournalFileBackend`` is what Optuna ships for shared filesystems.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import lightning as L
import torch

from prolit.config import (
    AtomVQVAEConfig,
    AtomVQVAETrainingConfig,
    CrossDockedConfig,
    HubDatasetConfig,
)
from prolit.data.atom_descriptors import AtomComplexDescriptorDataModule
from prolit.model.vqvae_module import AtomVQVAEModule
from prolit.provenance import write_manifest
from prolit.seeding import add_seed_argument, seed_from_args

#: Searched in log space around the hand-set values, wide enough to contain both
#: the tolerance-derived weights (~4-6 for every term) and the current ones.
_SPACE: dict[str, tuple[float, float]] = {
    "bond12": (0.2, 50.0),
    "bond13": (0.2, 50.0),
    "clash": (0.2, 50.0),
    "dmap": (0.05, 20.0),
}

#: ``coord`` is held at 1.0 rather than searched. Only ratios matter -- scaling
#: every geometry weight together is absorbed by the learning rate -- so fixing
#: one term removes a redundant dimension from an already sparse search.
_COORD = 1.0


def _objectives(metrics: dict[str, torch.Tensor]) -> tuple[float, float]:
    """(protein long-range MAE, ligand 1-2 MAE), both in Angstroms."""
    def get(name: str) -> float:
        v = metrics.get(name)
        if v is None:
            msg = f"{name} missing from callback_metrics; the diagnostic moved"
            raise KeyError(msg)
        return float(v)

    return get("val/atom_dmap_mae"), get("val/atom_bond12_mae")


def _run_trial(
    trial: Any,  # noqa: ANN401  optuna.Trial, imported lazily
    args: argparse.Namespace,
    dm: AtomComplexDescriptorDataModule,
) -> tuple[float, float]:
    weights = {
        name: trial.suggest_float(name, lo, hi, log=True)
        for name, (lo, hi) in _SPACE.items()
    }
    config = _training_config(args)
    config.atom.recon_weights = {
        **config.atom.recon_weights,
        "coord": _COORD,
        **weights,
    }
    module = AtomVQVAEModule(config)
    trainer = L.Trainer(
        max_epochs=args.trial_epochs,
        accelerator="auto",
        devices=1,
        precision=config.precision,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, dm)
    return _objectives(trainer.callback_metrics)


def _training_config(args: argparse.Namespace) -> AtomVQVAETrainingConfig:
    config = AtomVQVAETrainingConfig(atom=AtomVQVAEConfig())
    config.max_epochs = args.trial_epochs
    config.atom.codebook_size = args.codebook_size
    config.atom.bond_distance_loss = True
    config.atom.distance_map_loss = True
    config.atom.clash_floor = args.clash_floor
    config.atom.loss_balancing = "constrained"
    config.atom.ligand_source_weight = args.ligand_source_weight
    config.atom.ligand_ema_weight = args.ligand_source_weight
    # Every trial shares the study's seed, so a trial's result is attributable
    # to its weights and not to a draw. The front is a comparison between
    # weight settings; letting the seed move too would make it a comparison
    # between weight settings AND seeds, and the measured seed spread
    # (lDDT 0.016, SMILES 0.047) is wide enough to swamp it.
    config.seed = args.seed
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/descriptor_cache_allatom")
    )
    parser.add_argument("--trial-epochs", type=int, default=20)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument(
        "--time-budget-h",
        type=float,
        default=5.5,
        help="stop starting new trials after this many hours, so the job's "
        "walltime kills nothing mid-trial and the journal stays consistent",
    )
    parser.add_argument("--codebook-size", type=int, default=8192)
    parser.add_argument("--clash-floor", type=float, default=2.4)
    parser.add_argument("--ligand-source-weight", type=float, default=3.0)
    parser.add_argument(
        "--worker",
        type=int,
        default=0,
        help="index of this worker among the jobs sharing --study-dir. It "
        "offsets the SAMPLER's rng and nothing else: --seed still fixes model "
        "init for every trial, so a trial's result is attributable to its "
        "weights. Without the offset all workers would propose the same "
        "points, because seed_from_args pins the global rng the sampler draws "
        "from and the study only deduplicates after a trial has run.",
    )
    add_seed_argument(parser)
    args = parser.parse_args()
    seed_from_args(args)

    # Imported here, not at module scope: optuna pulls in sqlalchemy and this
    # module is imported by the layering test, which must stay cheap.
    import optuna  # noqa: PLC0415
    from optuna.storages import JournalStorage  # noqa: PLC0415
    from optuna.storages.journal import JournalFileBackend  # noqa: PLC0415

    torch.set_float32_matmul_precision("high")
    args.study_dir.mkdir(parents=True, exist_ok=True)
    # Written even though this is a search rather than a run: the front is only
    # reproducible with the command and the SHA that produced it. Not the
    # RecordProvenance callback -- that one reads its directory off a
    # ModelCheckpoint, and trials deliberately write no checkpoints.
    write_manifest(args.study_dir, seed=args.seed)

    config = _training_config(args)
    dm = AtomComplexDescriptorDataModule(
        config, CrossDockedConfig(), hub_config=_hub_config(), modality="both"
    )
    dm.cache_dir = args.cache_dir
    dm.prepare_data()
    dm.setup("fit")

    storage = JournalStorage(JournalFileBackend(str(args.study_dir / "study.log")))
    study = optuna.create_study(
        study_name="vqvae_geometry_weights",
        storage=storage,
        directions=["minimize", "minimize"],
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed + 1000 * args.worker),
    )
    deadline = time.monotonic() + args.time_budget_h * 3600.0
    study.optimize(
        lambda t: _run_trial(t, args, dm),
        n_trials=args.n_trials,
        timeout=max(0.0, deadline - time.monotonic()),
        catch=(RuntimeError, ValueError),
    )
    _dump_front(study, args.study_dir)


def _hub_config() -> HubDatasetConfig:
    hub = HubDatasetConfig()
    hub.exclude_eval_pdbs = True
    return hub


def _dump_front(study: Any, out_dir: Path) -> None:  # noqa: ANN401
    """Write the Pareto front so the next job does not have to re-open Optuna."""
    front = [
        {
            "number": t.number,
            "dmap_mae": t.values[0],
            "bond12_mae": t.values[1],
            "weights": {**t.params, "coord": _COORD},
        }
        for t in study.best_trials
    ]
    front.sort(key=lambda r: r["dmap_mae"])
    (out_dir / "pareto_front.json").write_text(json.dumps(front, indent=2))
    print(f"[tune] {len(study.trials)} trials, {len(front)} on the front")
    for row in front:
        print(
            f"  dmap {row['dmap_mae']:.4f}  bond12 {row['bond12_mae']:.4f}"
            f"  {row['weights']}"
        )


if __name__ == "__main__":
    main()

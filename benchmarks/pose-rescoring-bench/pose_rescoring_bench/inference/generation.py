"""Pocket-conditioned generation inference (orchestration) + sbdd-bench evaluation.

The all-atom token->3D->refine generator is model/decoder code that lives in the
source repo (``scripts/generate_ligands_3d.py``: AtomVQVAE + LM + pose refiner).
The paper's driver was an ephemeral scratchpad variant of it, so we treat the
in-repo all-atom generator as the black-box "model" and drive it by subprocess,
then run the sbdd-bench harness to score the generated molecules and consume its
per-molecule dump. Both wrappers live here; no eval code is added to the source
repo.

FIRST-RUN NOTE: the generator's pocket selection / output layout should be
confirmed on the first GPU run and the argument mapping adjusted via
``extra_args`` if needed — this is the one task whose inference could not be
validated offline (no reusable all-atom entrypoint besides the script).
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from pose_rescoring_bench.config import GenerationConfig, PathsConfig
    from pose_rescoring_bench.variants import GenerationCkpts

logger = logging.getLogger(__name__)


def generate(
    ckpts: GenerationCkpts,
    paths: PathsConfig,
    cfg: GenerationConfig,
    out_dir: Path,
    extra_args: list[str] | None = None,
) -> Path:
    """Run the source repo's all-atom 3D generator for one variant (subprocess).

    Writes generated molecules under ``out_dir``. Requires a GPU and the source
    repo's environment; intended for qsub.
    """
    if ckpts.lm is None or (ckpts.vqvae is None and not ckpts.is_separate):
        msg = "generation variant is missing lm/vqvae checkpoints"
        raise ValueError(msg)
    out_dir.mkdir(parents=True, exist_ok=True)
    script = paths.source_repo / "scripts" / "generate_ligands_3d.py"
    cmd = [
        "python",
        str(script),
        "--lm-ckpt",
        str(paths.ckpt(ckpts.lm)),
        "--codebook-size",
        str(ckpts.codebook_size),
        "--temperature",
        str(cfg.temperature),
        "--top-p",
        str(cfg.top_p),
        "--num-samples",
        str(cfg.n_samples),
        "--seed",
        str(cfg.seed),
        "--out-dir",
        str(out_dir),
    ]
    if ckpts.is_separate:
        # Separate arm: pocket encoded by the protein-only VQ, ligand decoded by
        # the ligand-only VQ over a combined 2*codebook-size space (SeparateVQVAE).
        if (
            ckpts.protein_vqvae is None
            or ckpts.protein_norm is None
            or ckpts.ligand_vqvae is None
            or ckpts.ligand_norm is None
        ):
            msg = "separate generation variant is missing protein/ligand ckpts or norms"
            raise ValueError(msg)
        cmd += [
            "--separate-protein-ckpt",
            str(paths.ckpt(ckpts.protein_vqvae)),
            "--separate-protein-norm",
            str(paths.ckpt(ckpts.protein_norm)),
            "--separate-ligand-ckpt",
            str(paths.ckpt(ckpts.ligand_vqvae)),
            "--separate-ligand-norm",
            str(paths.ckpt(ckpts.ligand_norm)),
        ]
    else:
        # Joint arm: single combined all-atom VQ (the generator's --all-atom path).
        if ckpts.vqvae is None:
            msg = "joint generation variant is missing the vqvae checkpoint"
            raise ValueError(msg)
        cmd += ["--all-atom", "--vqvae-ckpt", str(paths.ckpt(ckpts.vqvae))]
    if cfg.use_refiner and ckpts.refiner is not None:
        cmd += ["--refine-ckpt", str(paths.ckpt(ckpts.refiner))]
    cmd += extra_args or []
    logger.info("generating: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(paths.source_repo))  # noqa: S603
    return out_dir


def evaluate_with_sbdd(
    paths: PathsConfig,
    models: list[str],
    dock_modes: tuple[str, ...] = ("score", "min"),
    extra_args: list[str] | None = None,
) -> None:
    """Score generated molecules with the sbdd-bench harness (subprocess).

    Produces sbdd-bench's per-molecule/per-model outputs, which
    :mod:`pose_rescoring_bench.baselines.sbdd_gen` then collects into this repo's dumps.
    """
    script = paths.sbdd_bench_repo / "scripts" / "run_evaluation.py"
    cmd = [
        "python",
        str(script),
        "--models",
        *models,
        "--dock-modes",
        *dock_modes,
        *(extra_args or []),
    ]
    logger.info("sbdd-bench eval: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(paths.sbdd_bench_repo))  # noqa: S603


# ---------------------------------------------------------------------------
# CrossDocked2020 100-pocket path: drive the sbdd-bench "own" adapter (per-target
# generate_ligands_for_target.py) over the prepared 100-pocket target set, then
# score with the same harness that runs DiffSBDD/TargetDiff/DiffGui. Selecting
# our joint vs separate tokenizer is done through the adapter's SBDD_OWN_* env
# contract (see sbdd_bench/adapters/own.py) — no per-variant code in sbdd-bench.
# ---------------------------------------------------------------------------


def _own_env(
    ckpts: GenerationCkpts, paths: PathsConfig, cfg: GenerationConfig
) -> dict[str, str]:
    """Environment overrides selecting our variant in the sbdd-bench own adapter."""
    if ckpts.lm is None:
        msg = "generation variant is missing the lm checkpoint"
        raise ValueError(msg)
    env = dict(os.environ)
    env["SBDD_OWN_LM_CKPT"] = str(paths.ckpt(ckpts.lm))
    env["SBDD_OWN_CODEBOOK_SIZE"] = str(ckpts.codebook_size)
    if ckpts.is_separate:
        if (
            ckpts.protein_vqvae is None
            or ckpts.protein_norm is None
            or ckpts.ligand_vqvae is None
            or ckpts.ligand_norm is None
        ):
            msg = "separate generation variant is missing protein/ligand ckpts or norms"
            raise ValueError(msg)
        env["SBDD_OWN_MODE"] = "separate"
        env["SBDD_OWN_SEP_PROTEIN_CKPT"] = str(paths.ckpt(ckpts.protein_vqvae))
        env["SBDD_OWN_SEP_PROTEIN_NORM"] = str(paths.ckpt(ckpts.protein_norm))
        env["SBDD_OWN_SEP_LIGAND_CKPT"] = str(paths.ckpt(ckpts.ligand_vqvae))
        env["SBDD_OWN_SEP_LIGAND_NORM"] = str(paths.ckpt(ckpts.ligand_norm))
    else:
        if ckpts.vqvae is None:
            msg = "joint generation variant is missing the vqvae checkpoint"
            raise ValueError(msg)
        env["SBDD_OWN_MODE"] = "allatom"
        env["SBDD_OWN_VQVAE_CKPT"] = str(paths.ckpt(ckpts.vqvae))
        env["SBDD_OWN_NORM_STATS"] = str(paths.norm_stats)
    if cfg.use_refiner and ckpts.refiner is not None:
        env["SBDD_OWN_REFINE_CKPT"] = str(paths.ckpt(ckpts.refiner))
    return env


def generate_own_crossdocked(  # noqa: PLR0913
    ckpts: GenerationCkpts,
    paths: PathsConfig,
    cfg: GenerationConfig,
    *,
    index: Path,
    out_root: Path,
    n_samples: int,
    extra_args: list[str] | None = None,
) -> None:
    """Generate ligands for the 100-pocket set with our variant (subprocess, GPU).

    Runs sbdd-bench ``scripts/run_generation.py --models own`` with the SBDD_OWN_*
    env selecting this variant; SDFs land under ``out_root/own/<target>/``.
    """
    env = _own_env(ckpts, paths, cfg)
    script = paths.sbdd_bench_repo / "scripts" / "run_generation.py"
    cmd = [
        str(paths.bench_python),
        str(script),
        "--models",
        "own",
        "--index",
        str(index),
        "--n-samples",
        str(n_samples),
        "--out-dir",
        str(out_root),
        *(extra_args or []),
    ]
    logger.info("crossdocked gen (%s): %s", env.get("SBDD_OWN_MODE"), " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(paths.sbdd_bench_repo), env=env)  # noqa: S603


def evaluate_own_crossdocked(  # noqa: PLR0913
    paths: PathsConfig,
    *,
    index: Path,
    out_root: Path,
    results_dir: Path,
    dock_modes: tuple[str, ...] = ("score", "min", "dock"),
    extra_args: list[str] | None = None,
) -> None:
    """Score our 100-pocket generations with the sbdd-bench harness (subprocess).

    Writes ``results_dir/{per_model.csv,per_target.csv,per_molecule.parquet}`` in
    the exact layout ``pose_rescoring_bench.report`` reads for
    ``results/generation/<variant>``.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    script = paths.sbdd_bench_repo / "scripts" / "run_evaluation.py"
    cmd = [
        str(paths.bench_python),
        str(script),
        "--models",
        "own",
        "--index",
        str(index),
        "--out-dir",
        str(out_root),
        "--results",
        str(results_dir),
        "--dock-modes",
        *dock_modes,
        *(extra_args or []),
    ]
    logger.info("crossdocked eval: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(paths.sbdd_bench_repo))  # noqa: S603

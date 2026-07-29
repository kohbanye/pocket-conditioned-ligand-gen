"""Central path configuration for the benchmark.

Every path can be overridden with an environment variable so the bench runs the
same on a login node, a GPU node, or someone else's checkout.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# This bench's own directory (its weights/, data/, outputs/, results/ live here).
REPO_ROOT = Path(__file__).resolve().parent.parent
# The monorepo root: baseline sources and the ProLIT model package are shared
# across all benches, so they sit one level up rather than per-bench.
MONOREPO_ROOT = REPO_ROOT.parent.parent

THIRD_PARTY = MONOREPO_ROOT / "third_party"
ESM_REPO = THIRD_PARTY / "esm"
FOLDTOKEN_REPO = THIRD_PARTY / "FoldToken_open"
# ProLIT itself is no longer a submodule -- it is this repository.
OWN_MODEL_REPO = MONOREPO_ROOT
TOKEN_MOL_REPO = THIRD_PARTY / "token-mol"
# ConfSeq (Xiong et al., Nat Mach Intell 2026): rule-based, no weights needed.
CONFSEQ_REPO = THIRD_PARTY / "ConfSeq"
# Bio2Token (arXiv 2410.19110): all-atom Mamba+FSQ autoencoder, weights in-repo.
# mamba-ssm pins an exact torch build, so it gets a dedicated venv like FoldToken.
BIO2TOKEN_REPO = THIRD_PARTY / "bio2token"
BIO2TOKEN_CKPT_DIR = BIO2TOKEN_REPO / "checkpoints" / "bio2token"
_BIO2TOKEN_VENV_PY = REPO_ROOT / ".venv-bio2token" / "bin" / "python"
BIO2TOKEN_PYTHON = os.environ.get(
    "PLBENCH_BIO2TOKEN_PYTHON",
    str(_BIO2TOKEN_VENV_PY) if _BIO2TOKEN_VENV_PY.exists() else "python",
)

WEIGHTS_DIR = Path(os.environ.get("PLBENCH_WEIGHTS_DIR", REPO_ROOT / "weights"))
DATA_DIR = Path(os.environ.get("PLBENCH_DATA_DIR", REPO_ROOT / "data"))
OUTPUTS_DIR = Path(os.environ.get("PLBENCH_OUTPUTS_DIR", REPO_ROOT / "outputs"))
RESULTS_DIR = Path(os.environ.get("PLBENCH_RESULTS_DIR", REPO_ROOT / "results"))

# --- FoldToken ---------------------------------------------------------------
FOLDTOKEN_FT4_DIR = WEIGHTS_DIR / "foldtoken" / "model_zoom" / "FT4"
FOLDTOKEN_CONFIG = FOLDTOKEN_FT4_DIR / "config.yaml"
FOLDTOKEN_CKPT = FOLDTOKEN_FT4_DIR / "ckpt.pth"
# FoldToken's deps are old; run it through a dedicated uv venv (no conda).
# Created by: uv venv --python 3.10 .venv-foldtoken && (see scripts/setup_foldtoken_env.sh)
_FOLDTOKEN_VENV_PY = REPO_ROOT / ".venv-foldtoken" / "bin" / "python"
FOLDTOKEN_PYTHON = os.environ.get(
    "PLBENCH_FOLDTOKEN_PYTHON",
    str(_FOLDTOKEN_VENV_PY) if _FOLDTOKEN_VENV_PY.exists() else "python",
)

# --- ProLIT ------------------------------------------------------------------
# ProLIT lives in this monorepo, but it has its own uv venv: this bench cannot
# share one with it (ESM3 pins a fork of transformers that would downgrade the
# transformers ProLIT's language models need).
OWN_MODEL_WORKDIR = Path(
    os.environ.get("PLBENCH_OWN_MODEL_WORKDIR", MONOREPO_ROOT)
)
OWN_MODEL_PYTHON = os.environ.get(
    "PLBENCH_OWN_MODEL_PYTHON", str(OWN_MODEL_WORKDIR / ".venv" / "bin" / "python")
)

# Pocket atoms and ligand atoms share one 33-D descriptor so a single codebook
# can cover both. Checkpoints and descriptor caches are read straight out of the
# model's run directories -- there is one per ablation arm and they are far too
# large to symlink individually.
OWN_VQ_RUNS_DIR = Path(
    os.environ.get("PLBENCH_OWN_VQ_RUNS_DIR", OWN_MODEL_WORKDIR / "pocket-ligand-vqvae")
)
OWN_ALLATOM_CACHE = Path(
    os.environ.get(
        "PLBENCH_OWN_ALLATOM_CACHE",
        OWN_MODEL_WORKDIR / "data" / "descriptor_cache_allatom",
    )
)
# Ligand descriptors built in the ligand's OWN canonical frame (the
# single-modality-tokenizer ablation) rather than the shared pocket frame.
OWN_LOCALFRAME_CACHE = Path(
    os.environ.get(
        "PLBENCH_OWN_LOCALFRAME_CACHE",
        OWN_MODEL_WORKDIR / "data" / "descriptor_cache_ligand_localframe",
    )
)

# Parsed receptors, pickled and shared across arms and runs. Re-parsing a CASP
# protein costs ~10 s; every all-atom arm runs as its own process, so without
# this the same proteins are parsed once per arm.
RECEPTOR_CACHE = Path(
    os.environ.get("PLBENCH_RECEPTOR_CACHE", str(OUTPUTS_DIR / "receptor_cache"))
)

# --- OpenBabel ---------------------------------------------------------------
# Used to perceive ligand bond orders when converting CASP ligand PDBs to SDF.
# RDKit's PDB reader does not perceive bond orders at all (every bond comes back
# SINGLE), which silently breaks any chemistry-aware metric downstream.
OBABEL = os.environ.get(
    "PLBENCH_OBABEL",
    shutil.which("obabel") or "/home/5/uq02055/usr/app/babel/bin/obabel",
)

# --- ESM3 --------------------------------------------------------------------
# Structure encoder/decoder weights are fetched from HuggingFace
# (biohub/esm3-sm-open-v1) into the HF cache on first use.
ESM3_HF_REPO = "biohub/esm3-sm-open-v1"


def ensure_dirs() -> None:
    for d in (WEIGHTS_DIR, DATA_DIR, OUTPUTS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

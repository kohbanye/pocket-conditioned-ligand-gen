"""Central path configuration for the benchmark.

Every path can be overridden with an environment variable so the bench runs the
same on a login node, a GPU node, or someone else's checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

THIRD_PARTY = REPO_ROOT / "third_party"
ESM_REPO = THIRD_PARTY / "esm"
FOLDTOKEN_REPO = THIRD_PARTY / "FoldToken_open"
OWN_MODEL_REPO = THIRD_PARTY / "pocket-conditioned-ligand-gen"
TOKEN_MOL_REPO = THIRD_PARTY / "token-mol"

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

# --- Own pocket-ligand VQ-VAE ------------------------------------------------
# Source comes from the submodule, but the trained weights + descriptor cache
# live in a separate working copy and are symlinked into weights/ and data/.
OWN_VQVAE_CKPT = WEIGHTS_DIR / "own_vqvae" / "vqvae.ckpt"
OWN_DESCRIPTOR_CACHE = DATA_DIR / "own_descriptor_cache"
# The own model has its own uv venv with the exact deps + cache wiring.
OWN_MODEL_WORKDIR = Path(
    os.environ.get(
        "PLBENCH_OWN_MODEL_WORKDIR",
        "/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen",
    )
)
OWN_MODEL_PYTHON = os.environ.get(
    "PLBENCH_OWN_MODEL_PYTHON", str(OWN_MODEL_WORKDIR / ".venv" / "bin" / "python")
)

# --- ESM3 --------------------------------------------------------------------
# Structure encoder/decoder weights are fetched from HuggingFace
# (biohub/esm3-sm-open-v1) into the HF cache on first use.
ESM3_HF_REPO = "biohub/esm3-sm-open-v1"


def ensure_dirs() -> None:
    for d in (WEIGHTS_DIR, DATA_DIR, OUTPUTS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

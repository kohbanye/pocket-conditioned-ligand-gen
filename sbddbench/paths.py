"""Central path / tool / environment configuration for the SBDD benchmark.

Every path can be overridden with an environment variable so the bench runs the
same on a login node, a GPU node, or someone else's checkout. The guiding split:

* **This (bench) env** only ever *evaluates* — it reads ``generated.sdf`` files
  and scores them with RDKit + AutoDock Vina + PoseBusters. It imports no
  generative model.
* **Each model** *generates* in its own interpreter (``*_PYTHON`` below), driven
  as a subprocess. Their deps are heavy and mutually incompatible, so they never
  share an env with each other or with the bench.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- model source (git submodules) -------------------------------------------
THIRD_PARTY = REPO_ROOT / "third_party"
OWN_MODEL_REPO = THIRD_PARTY / "pocket-conditioned-ligand-gen"
DIFFSBDD_REPO = THIRD_PARTY / "DiffSBDD"
TARGETDIFF_REPO = THIRD_PARTY / "targetdiff"
DIFFGUI_REPO = THIRD_PARTY / "DiffGui"

# --- bench-managed directories (all git-ignored) -----------------------------
WEIGHTS_DIR = Path(os.environ.get("SBDD_WEIGHTS_DIR", REPO_ROOT / "weights"))
DATA_DIR = Path(os.environ.get("SBDD_DATA_DIR", REPO_ROOT / "data"))
TARGETS_DIR = Path(os.environ.get("SBDD_TARGETS_DIR", DATA_DIR / "targets"))
OUTPUTS_DIR = Path(os.environ.get("SBDD_OUTPUTS_DIR", REPO_ROOT / "outputs"))
RESULTS_DIR = Path(os.environ.get("SBDD_RESULTS_DIR", REPO_ROOT / "results"))

# --- system tools (live outside any venv) ------------------------------------
# AutoDock Vina + Open Babel power docking and ligand prep; ADFRsuite's
# prepare_receptor builds receptor pdbqt. Resolved from PATH, then known
# install locations on this machine, overridable via env.
def _tool(env: str, *candidates: str) -> str:
    if env in os.environ:
        return os.environ[env]
    name = candidates[0].rsplit("/", 1)[-1]
    found = shutil.which(name)
    if found:
        return found
    for c in candidates:
        if Path(c).exists():
            return c
    return name  # last resort: hope it is on PATH at call time


VINA = _tool("SBDD_VINA", "/home/5/uq02055/.local/bin/vina", "vina")
OBABEL = _tool("SBDD_OBABEL", "/home/5/uq02055/usr/app/babel/bin/obabel", "obabel")
PREPARE_RECEPTOR = _tool(
    "SBDD_PREPARE_RECEPTOR",
    "/home/5/uq02055/usr/app/ADFRsuite/bin/prepare_receptor",
    "prepare_receptor",
)

# ============================================================================
# Per-model generation environments + checkpoints.
# Each model is run by its *own* interpreter as a subprocess. The default
# interpreter paths point at envs created by scripts/setup_envs.sh; override
# any of them with the matching env var.
# ============================================================================

# --- pocket-conditioned-ligand-gen (Ours) ------------------------------------
# Source is the submodule, but the trained weights + descriptor cache (the
# normalization stats) live in a separate working copy and are symlinked into
# weights/ and data/. Generation runs in that working copy's uv venv.
OWN_MODEL_WORKDIR = Path(
    os.environ.get(
        "SBDD_OWN_WORKDIR",
        "/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen",
    )
)
OWN_PYTHON = os.environ.get(
    "SBDD_OWN_PYTHON", str(OWN_MODEL_WORKDIR / ".venv" / "bin" / "python")
)
# Symlinked into weights/ by scripts/fetch_weights.py --own.
OWN_LM_CKPT = WEIGHTS_DIR / "own" / "lm.ckpt"
OWN_VQVAE_CKPT = WEIGHTS_DIR / "own" / "vqvae.ckpt"
OWN_DESCRIPTOR_CACHE = DATA_DIR / "own_descriptor_cache"
# Default sources in the working copy (overridable in fetch_weights.py).
OWN_LM_CKPT_SRC = OWN_MODEL_WORKDIR / "pocket-ligand-lm/g79let5b/checkpoints/lm-e09-vl1.4088.ckpt"
OWN_VQVAE_CKPT_SRC = (
    OWN_MODEL_WORKDIR
    / "pocket-ligand-vqvae/3dvcbp0h/checkpoints/vqvae-epoch=99-val/ligand_coord=0.1501.ckpt"
)
OWN_DESCRIPTOR_CACHE_SRC = OWN_MODEL_WORKDIR / "data/descriptor_cache_v4"

# --- DiffSBDD (Schneuing et al. 2024) ----------------------------------------
DIFFSBDD_PYTHON = os.environ.get(
    "SBDD_DIFFSBDD_PYTHON", str(REPO_ROOT / "envs" / "diffsbdd" / "bin" / "python")
)
# CrossDocked conditional full-atom checkpoint (the published SBDD model).
DIFFSBDD_CKPT = Path(
    os.environ.get("SBDD_DIFFSBDD_CKPT", WEIGHTS_DIR / "diffsbdd" / "crossdocked_fullatom_cond.ckpt")
)

# --- TargetDiff (Guan et al. 2023) -------------------------------------------
TARGETDIFF_PYTHON = os.environ.get(
    "SBDD_TARGETDIFF_PYTHON", str(REPO_ROOT / "envs" / "targetdiff" / "bin" / "python")
)
TARGETDIFF_CKPT = Path(
    os.environ.get("SBDD_TARGETDIFF_CKPT", WEIGHTS_DIR / "targetdiff" / "pretrained_targetdiff.pt")
)

# --- DiffGui (Hu et al. 2024) ------------------------------------------------
DIFFGUI_PYTHON = os.environ.get(
    "SBDD_DIFFGUI_PYTHON", str(REPO_ROOT / "envs" / "diffgui" / "bin" / "python")
)
DIFFGUI_CKPT = Path(
    os.environ.get("SBDD_DIFFGUI_CKPT", WEIGHTS_DIR / "diffgui" / "diffgui.pt")
)


def ensure_dirs() -> None:
    for d in (WEIGHTS_DIR, DATA_DIR, TARGETS_DIR, OUTPUTS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

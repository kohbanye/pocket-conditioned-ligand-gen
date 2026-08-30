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
from pathlib import Path

# This bench's own directory (its weights/, data/, outputs/, results/ live here).
REPO_ROOT = Path(__file__).resolve().parent.parent
# The monorepo root: baseline sources and the ProLIT model package are shared
# across all benches, so they sit one level up rather than per-bench.
MONOREPO_ROOT = REPO_ROOT.parent.parent

# --- model source (git submodules) -------------------------------------------
THIRD_PARTY = MONOREPO_ROOT / "third_party"
# ProLIT itself is no longer a submodule -- it is this repository.
OWN_MODEL_REPO = MONOREPO_ROOT
DIFFSBDD_REPO = THIRD_PARTY / "DiffSBDD"
FLOWR_REPO = THIRD_PARTY / "flowr"
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
# prepare_receptor builds the receptor pdbqt. None is a Python package, so where
# they live is a property of the machine.
#
# Resolution is delegated to :mod:`prolit.external_tools` so a job that points
# at Open Babel once points at it for the whole run. Having a second, private
# lookup here was a real trap: ``PROLIT_OBABEL`` satisfied the library and the
# benchmark still went looking for a bare ``obabel``, so two batches of scoring
# jobs died several minutes in, past the point where the failure was cheap. The
# ``SBDD_*`` names still win where they are set, so an existing environment
# keeps working.
def _tool(env: str, name: str) -> str:
    from prolit.external_tools import find_tool

    return os.environ.get(env) or find_tool(name) or name


VINA = _tool("SBDD_VINA", "vina")
OBABEL = _tool("SBDD_OBABEL", "obabel")
PREPARE_RECEPTOR = _tool("SBDD_PREPARE_RECEPTOR", "prepare_receptor")

# ============================================================================
# Per-model generation environments + checkpoints.
# Each model is run by its *own* interpreter as a subprocess. The default
# interpreter paths point at envs created by scripts/setup_envs.sh; override
# any of them with the matching env var.
# ============================================================================

# --- ProLIT (Ours) -----------------------------------------------------------
# ProLIT lives in this monorepo. Its trained weights and descriptor caches are
# not in git; they are symlinked into weights/ and data/ by fetch_weights.py.
OWN_MODEL_WORKDIR = Path(os.environ.get("SBDD_OWN_WORKDIR", MONOREPO_ROOT))
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
# FLOWR (Cremer et al. 2025). Trained on SPINDR rather than CrossDocked -- the
# test set and the scorer are what the comparison holds fixed, not the training
# data, and the paper has to say so.
FLOWR_PYTHON = os.environ.get(
    "SBDD_FLOWR_PYTHON", "/gs/bs/tga-ohuelab/sakano/envs/flowr/bin/python"
)
FLOWR_CKPT = Path(
    os.environ.get("SBDD_FLOWR_CKPT", WEIGHTS_DIR / "flowr" / "flowr_noHs.ckpt")
)

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

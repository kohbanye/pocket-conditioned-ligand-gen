"""Central path / tool configuration for the affinity benchmark.

Every path can be overridden with an environment variable so the bench runs the
same on a login node, a GPU node, or someone else's checkout. The split matches
the other benches:

* **This (bench) env** only ever *evaluates* -- it reads per-complex dumps and
  computes correlations. It imports no baseline.
* **Each baseline** scores in its own interpreter (``*_PYTHON`` below), driven as
  a subprocess. Their dependencies are mutually incompatible -- GenScore pins a
  DGL build, GIGN pins torch 1.10 with matching torch_sparse, OnionNet-2 pins
  TensorFlow 2.3 -- so they never share an environment with each other or with
  this bench.

Baseline sources are git submodules under ``third_party/`` and are pinned to the
commit each published number was measured at. Nothing here reads the old
``../baselines`` checkout; ``GENSCORE_PYTHON`` still falls back to the conda env
that lives there, because rebuilding a working DGL environment is not free.
"""

from __future__ import annotations

import os
from pathlib import Path

# benchmarks/affinity-bench/affinity_bench/paths.py -> the bench dir.
REPO_ROOT = Path(__file__).resolve().parent.parent
# The monorepo root: baseline sources are shared across benches, so they sit at
# the top rather than per-bench.
MONOREPO_ROOT = REPO_ROOT.parent.parent

# --- baseline sources (git submodules) ---------------------------------------
THIRD_PARTY = MONOREPO_ROOT / "third_party"
GENSCORE_REPO = Path(os.environ.get("AFFINITY_GENSCORE_REPO", THIRD_PARTY / "GenScore"))
RTMSCORE_REPO = Path(os.environ.get("AFFINITY_RTMSCORE_REPO", THIRD_PARTY / "RTMScore"))
GIGN_REPO = Path(os.environ.get("AFFINITY_GIGN_REPO", THIRD_PARTY / "GIGN"))
SFCNN_REPO = Path(os.environ.get("AFFINITY_SFCNN_REPO", THIRD_PARTY / "Sfcnn"))
PLANET_REPO = Path(os.environ.get("AFFINITY_PLANET_REPO", THIRD_PARTY / "PLANET"))
ONIONNET2_REPO = Path(
    os.environ.get("AFFINITY_ONIONNET2_REPO", THIRD_PARTY / "OnionNet-2")
)

# --- bench-managed directories (git-ignored) ---------------------------------
WEIGHTS_DIR = Path(os.environ.get("AFFINITY_WEIGHTS_DIR", REPO_ROOT / "weights"))
WORK_DIR = Path(os.environ.get("AFFINITY_WORK_DIR", REPO_ROOT / "work"))
RESULTS_DIR = Path(os.environ.get("AFFINITY_RESULTS_DIR", REPO_ROOT / "results"))

# --- evaluation data ---------------------------------------------------------
CASF_DIR = Path(
    os.environ.get("AFFINITY_CASF_DIR", MONOREPO_ROOT / "data" / "casf2016")
)
CASF_CORESET = CASF_DIR / "coreset"
CASF_LABELS = CASF_DIR / "power_scoring" / "CoreSet.dat"

# --- per-baseline interpreters -----------------------------------------------
def _venv_python(name: str, env_var: str, legacy: Path | None = None) -> str:
    """A baseline's interpreter: env var, else bench venv, else legacy, else PATH."""
    override = os.environ.get(env_var)
    if override:
        return override
    candidate = REPO_ROOT / f".venv-{name}" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    if legacy is not None and legacy.exists():
        return str(legacy)
    return "python"


#: The pre-submodule checkout. GenScore's DGL build still lives there and works;
#: it is a fallback, not a location anything new should be written to.
_LEGACY_BASELINES = MONOREPO_ROOT.parent / "baselines"

GIGN_PYTHON = _venv_python(
    "gign",
    "AFFINITY_GIGN_PYTHON",
    _LEGACY_BASELINES / "envs" / "gign" / "bin" / "python",
)
ONIONNET2_PYTHON = _venv_python(
    "onionnet2", "AFFINITY_ONIONNET2_PYTHON",
    _LEGACY_BASELINES / "envs" / "onionnet2" / "bin" / "python",
)
GENSCORE_PYTHON = _venv_python(
    "genscore", "AFFINITY_GENSCORE_PYTHON",
    _LEGACY_BASELINES / "envs" / "genscore" / "bin" / "python",
)
#: Sfcnn has no pre-submodule checkout: its env was built by
#: scripts/setup_sfcnn_env.sh and lives only under the bench.
SFCNN_PYTHON = _venv_python("sfcnn", "AFFINITY_SFCNN_PYTHON")
#: PLANET ships its own pyproject and uv config, so its venv lives in the
#: checkout rather than under the bench like the others.
PLANET_PYTHON = os.environ.get(
    "AFFINITY_PLANET_PYTHON", str(PLANET_REPO / ".venv-planet" / "bin" / "python")
)

# --- baseline weights --------------------------------------------------------
#: OnionNet-2's released model. Upstream tracks it with git-lfs, so a submodule
#: checkout without git-lfs leaves a ~130-byte pointer where 885 MB should be.
#: It is fetched from Zenodo instead (scripts/fetch_weights.py) and the adapter
#: checks the HDF5 magic -- handing Keras a pointer file fails deep inside the
#: loader with an error that says nothing about what went wrong.
ONIONNET2_MODEL = Path(
    os.environ.get("AFFINITY_ONIONNET2_MODEL",
                   WEIGHTS_DIR / "onionnet2" / "62shell_saved-model.h5")
)
ONIONNET2_ZENODO = (
    "https://zenodo.org/api/records/10728089/files/62shell_saved-model.h5/content"
)
#: Small enough that upstream tracks it normally; read straight from the submodule.
ONIONNET2_SCALER = ONIONNET2_REPO / "models" / "train_scaler.scaler"

#: GIGN ships three trained checkpoints (repeat0/1/2) inside its repository.
GIGN_MODEL_DIR = GIGN_REPO / "GIGN" / "model"


def gign_checkpoints() -> list[Path]:
    """All three released GIGN checkpoints, in a stable order.

    All of them are scored and reported: choosing the best by its CASF result
    would be selection on the test set, which is the rule our own arms are held
    to.
    """
    return sorted(GIGN_MODEL_DIR.glob("*/model/*.pt"))


# --- system tools ------------------------------------------------------------
def _tool(env: str, name: str) -> str:
    """Delegate to the library's resolver so one setting points the whole run."""
    from prolit.external_tools import find_tool  # noqa: PLC0415

    return os.environ.get(env) or find_tool(name) or name


#: OnionNet-2 and GIGN both need ligand mol2 -> pdb before their parsers see it.
OBABEL = _tool("AFFINITY_OBABEL", "obabel")


def ensure_dirs() -> None:
    for d in (WEIGHTS_DIR, WORK_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

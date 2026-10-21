"""Where each pose baseline's source, interpreter and weights live.

``config.PathsConfig`` covers what *our* arms need -- checkpoints, CASF inputs,
a results tree -- and is instantiated per run. This module covers the other side
of the comparison: the published baselines, which cannot be configured per run
because each one only works inside its own interpreter.

* **This (bench) env** only ever *evaluates*: it reads per-pose dumps and
  computes docking power. It imports no baseline.
* **Each baseline** scores in its own interpreter (``*_PYTHON`` below), driven as
  a subprocess. RTMScore pins a DGL build and GenScore a PyTorch-Geometric one,
  so they share an environment with neither each other nor this bench.

Baseline sources are the ``third_party/`` submodules, pinned to the commit each
number is measured at. The interpreters still fall back to the pre-submodule
checkout's conda environments: a working DGL build is expensive to rebuild, and
it is the *source* that has to be pinned, not the interpreter.

Every path can be overridden with an environment variable so the bench runs the
same on a login node, a GPU node, or someone else's checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

# benchmarks/pose-rescoring-bench/pose_rescoring_bench/paths.py -> the bench dir.
BENCH_ROOT = Path(__file__).resolve().parent.parent
# Baseline sources are shared across benches, so they sit at the monorepo top.
MONOREPO_ROOT = BENCH_ROOT.parent.parent

# --- baseline sources (git submodules) ---------------------------------------
THIRD_PARTY = MONOREPO_ROOT / "third_party"
RTMSCORE_REPO = Path(
    os.environ.get("RESCORING_BENCH_RTMSCORE_REPO", THIRD_PARTY / "RTMScore")
)
GENSCORE_REPO = Path(
    os.environ.get("RESCORING_BENCH_GENSCORE_REPO", THIRD_PARTY / "GenScore")
)

# --- bench-managed directories (git-ignored) ---------------------------------
WEIGHTS_DIR = Path(
    os.environ.get("RESCORING_BENCH_WEIGHTS_DIR", BENCH_ROOT / "weights")
)
WORK_DIR = Path(os.environ.get("RESCORING_BENCH_WORK_DIR", BENCH_ROOT / "work"))
RESULTS_DIR = Path(
    os.environ.get("RESCORING_BENCH_RESULTS_DIR", BENCH_ROOT / "results")
)

# --- evaluation data ---------------------------------------------------------
CASF_DIR = Path(
    os.environ.get("RESCORING_BENCH_CASF_DIR", MONOREPO_ROOT / "data" / "casf2016")
)
CASF_CORESET = CASF_DIR / "coreset"
#: One multi-pose ``<pdbid>_decoys.mol2`` and one ``<pdbid>_rmsd.dat`` per target.
CASF_DECOYS = CASF_DIR / "decoys_docking"
#: The docking-power target list. Distinct from ``power_scoring/CoreSet.dat``:
#: docking power is defined over the targets that ship a decoy stack.
CASF_DOCKING_LIST = CASF_DIR / "power_docking" / "CoreSet.dat"


# --- per-baseline interpreters -----------------------------------------------
def _venv_python(name: str, env_var: str, legacy: Path | None = None) -> str:
    """A baseline's interpreter: env var, else bench venv, else legacy, else PATH."""
    override = os.environ.get(env_var)
    if override:
        return override
    candidate = BENCH_ROOT / f".venv-{name}" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    if legacy is not None and legacy.exists():
        return str(legacy)
    return "python"


#: The pre-submodule checkout. Both DGL/PyG environments still live there and
#: work; it is a fallback, not a location anything new should be written to.
_LEGACY_BASELINES = MONOREPO_ROOT.parent / "baselines"

RTMSCORE_PYTHON = _venv_python(
    "rtmscore",
    "RESCORING_BENCH_RTMSCORE_PYTHON",
    _LEGACY_BASELINES / "envs" / "rtmscore" / "bin" / "python",
)
GENSCORE_PYTHON = _venv_python(
    "genscore",
    "RESCORING_BENCH_GENSCORE_PYTHON",
    _LEGACY_BASELINES / "envs" / "genscore" / "bin" / "python",
)

# --- baseline weights --------------------------------------------------------
#: Both repositories ship their released checkpoints in-tree, so there is
#: nothing to fetch. RTMScore ships three; model1 is the one its paper reports,
#: and scoring the other two is a knob rather than a default, because picking
#: among them by their CASF result would be selection on the test set.
RTMSCORE_CHECKPOINT = "rtmscore_model1.pth"

#: gnina ships no separate weights: the CNN ensemble is inside the binary. The
#: release asset is built against a newer glibc than this cluster has, so the
#: container is the artefact, and its digest is what pins the version.
GNINA_SIF = Path(
    os.environ.get("RESCORING_BENCH_GNINA_SIF", WEIGHTS_DIR / "gnina" / "gnina.sif")
)
GENSCORE_CHECKPOINT = "GatedGCN_ft_1.0_1.pth"


def rtmscore_checkpoints() -> list[Path]:
    """All three released RTMScore checkpoints, in a stable order."""
    return sorted((RTMSCORE_REPO / "trained_models").glob("rtmscore_model*.pth"))


# --- system tools ------------------------------------------------------------
def _tool(env: str, name: str) -> str:
    """Delegate to the library's resolver so one setting points the whole run."""
    from prolit.external_tools import find_tool  # noqa: PLC0415

    return os.environ.get(env) or find_tool(name) or name


#: Both DGL baselines read ligands as sdf; RDKit's mol2 reader fails silently on
#: some CASF decoy stacks, so the conversion goes through OpenBabel.
OBABEL = _tool("RESCORING_BENCH_OBABEL", "obabel")
VINA = _tool("RESCORING_BENCH_VINA", "vina")


def require_tool(resolved: str, name: str) -> None:
    """Fail now if ``name`` did not resolve to a path.

    ``_tool`` falls back to the bare name so an interactive run picks the tool
    up from ``PATH``. Inside a job there is no ``PATH`` to fall back to: job
    scripts deliberately do not source the interactive rc file (see
    ``jobs/lib.sh``), so a tool installed under ``$HOME`` is simply absent. An
    unresolved name then reaches ``subprocess`` once per target and fails 285
    times with a bare "No such file or directory" -- which is what happened on
    2026-09-17, and which costs a GPU-hour to learn. Checking once, up front,
    turns that into an immediate error naming the variable to set.
    """
    if resolved != name:
        return
    env_var = f"PROLIT_{name.upper()}"
    msg = (
        f"{name!r} did not resolve to a path. Set {env_var} (or "
        f"RESCORING_BENCH_{name.upper()}) to its full path -- a job's PATH does "
        "not include an interactive shell's additions."
    )
    raise FileNotFoundError(msg)


def ensure_dirs() -> None:
    for d in (WEIGHTS_DIR, WORK_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

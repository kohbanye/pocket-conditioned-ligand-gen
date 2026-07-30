#!/usr/bin/env bash
# Create the per-model generation environments under envs/<model>.
#
# Each external SBDD model pins its own (old, mutually incompatible) CUDA / PyG
# stack, so each gets an isolated env that the adapters call as a subprocess.
# The bench's own evaluation env is separate (uv sync at the repo root) and is
# NOT created here. Ours uses the working copy's existing uv venv.
#
# Usage:
#   sh scripts/setup_envs.sh diffsbdd          # one model
#   sh scripts/setup_envs.sh diffsbdd targetdiff diffgui
#   sh scripts/setup_envs.sh all
#
# Requires micromamba (or conda/mamba) on PATH. The upstream env files are
# exact-build exports pinned to CUDA 11.8; on newer drivers/GPUs you may need to
# relax torch / pytorch-{scatter,cluster,sparse} to a cu118 build that supports
# your card (cu118 covers H100/sm_90).
set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENVS_DIR="$REPO_ROOT/envs"
TP="$REPO_ROOT/third_party"
mkdir -p "$ENVS_DIR"

MM="$(command -v micromamba || command -v mamba || command -v conda || true)"
if [ -z "$MM" ]; then
  echo "ERROR: need micromamba/mamba/conda on PATH" >&2
  exit 1
fi
# micromamba needs a root prefix for its package cache.
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$REPO_ROOT/.mamba}"
mkdir -p "$MAMBA_ROOT_PREFIX"
# Note: the adapters set BABEL_LIBDIR/BABEL_DATADIR per-env at run time so the
# conda OpenBabel python bindings find their plugins (DiffSBDD needs this).

create_env () {  # name  env_file
  name="$1"; env_file="$2"
  prefix="$ENVS_DIR/$name"
  if [ -x "$prefix/bin/python" ]; then
    echo "[$name] already present at $prefix"; return
  fi
  echo "[$name] creating env from $env_file -> $prefix"
  "$MM" create -y -p "$prefix" -f "$env_file" || {
    echo "[$name] solve failed. Try relaxing the env file (drop exact build" \
         "strings / bump torch to a cu118 build for your GPU), then re-run." >&2
    return 1
  }
  # pytorch-lightning 1.x imports pkg_resources, removed in setuptools>=81.
  "$MM" install -y -p "$prefix" -c conda-forge "setuptools<81" >/dev/null 2>&1 || true
  echo "[$name] done: $prefix/bin/python"
}

want="$*"
[ -z "$want" ] && { echo "usage: sh scripts/setup_envs.sh [diffsbdd|targetdiff|diffgui|all]"; exit 1; }
[ "$want" = "all" ] && want="diffsbdd targetdiff diffgui"

# TargetDiff / DiffGui ship exact-build CUDA-11.6/py3.7 exports that no longer
# solve; envs_spec/ holds minimal specs on the proven cu118 stack instead.
for m in $want; do
  case "$m" in
    diffsbdd)   create_env diffsbdd   "$TP/DiffSBDD/environment.yaml" ;;
    targetdiff) create_env targetdiff "$REPO_ROOT/envs_spec/targetdiff.yaml" ;;
    diffgui)    create_env diffgui    "$REPO_ROOT/envs_spec/diffgui.yaml" ;;
    *) echo "unknown model: $m" >&2 ;;
  esac
done

cat <<EOF

Next:
  1. Fetch checkpoints:   python scripts/fetch_weights.py --all
  2. Generate (GPU node): python scripts/run_generation.py --models <model> --n-samples 100
  3. Evaluate (bench env): python scripts/run_evaluation.py --models <model>

Override an interpreter path with SBDD_<MODEL>_PYTHON if you keep an env elsewhere.
EOF

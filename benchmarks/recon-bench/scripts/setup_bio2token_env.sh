#!/bin/sh
# Build a dedicated uv venv for Bio2Token (no conda).
#
# Bio2Token pins torch 2.4.1+cu121 and needs mamba-ssm, whose CUDA kernels are
# built against an exact (torch, CUDA, cxx11-abi, cpython) tuple. That is far too
# specific to force on the bench env, so it gets its own venv and the adapter
# drives it through a subprocess -- the same arrangement FoldToken already uses.
#
# causal-conv1d and mamba-ssm are installed from the upstream prebuilt wheels;
# building them from source needs nvcc and takes the better part of an hour.
# --no-build-isolation keeps pip from re-resolving torch while building.
#
#   sh scripts/setup_bio2token_env.sh
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_ROOT/.venv-bio2token"
VPY="$VENV/bin/python"

uv venv --python 3.11 "$VENV"
uv pip install --python "$VPY" torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Prebuilt kernels matching torch 2.4 / cu12 / cp311. The cxx11abiFALSE variant
# is the one that matches the PyPI torch wheels.
CC_URL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
MS_URL="https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
uv pip install --python "$VPY" "$CC_URL" || echo "[warn] causal-conv1d wheel failed"
uv pip install --python "$VPY" "$MS_URL"

# Everything the reconstruction path actually imports. Training-only extras
# (mlflow, hydra-zen, nglview, jupyter) are deliberately left out.
uv pip install --python "$VPY" \
    setuptools biopython numpy pandas scipy pyyaml einops \
    "lightning>=2.4" hydra-zen python-box loguru dill \
    invariant-point-attention "transformers==4.44.2"

uv pip install --python "$VPY" --no-deps -e "$REPO_ROOT/third_party/bio2token"

"$VPY" -c "import mamba_ssm, torch; print('mamba-ssm', mamba_ssm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "[setup] bio2token venv ready at $VENV"

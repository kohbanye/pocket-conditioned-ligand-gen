#!/bin/sh
# Build a dedicated uv venv for FoldToken4 (no conda).
#
# FoldToken's upstream env is conda + CUDA 11.7; reconstruction only needs torch
# + PyG (scatter/cluster) + pytorch-lightning + a few small libs (no flash-attn,
# no openfold, no deepspeed — those are training/aux only). We use the cu118
# build so it runs on H100 (sm_90).
#
#   sh scripts/setup_foldtoken_env.sh
#
# Then run the benchmark with FoldToken (the bench auto-detects .venv-foldtoken):
#   HF_HUB_OFFLINE=1 uv run python scripts/run_reconstruction.py \
#       --models own_vqvae esm3 foldtoken --dataset casp16 --limit 50
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_ROOT/.venv-foldtoken"
VPY="$VENV/bin/python"

uv venv --python 3.10 "$VENV"
uv pip install --python "$VPY" torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
uv pip install --python "$VPY" torch_scatter torch_cluster \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
uv pip install --python "$VPY" \
    "torch-geometric==2.3.1" "pytorch-lightning==1.9.0" "torchmetrics<1.3" \
    "numpy<2" "setuptools<81" scipy einops omegaconf tqdm biotite biopython \
    ml-collections easydict

echo "FoldToken env ready at $VENV"

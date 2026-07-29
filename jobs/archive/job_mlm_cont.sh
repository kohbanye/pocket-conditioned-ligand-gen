#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -N mlm_cont

# Continue the ESM3-style complex-token MLM backbone (~99M) from the best
# 1-epoch checkpoint (j90rlrgm, masked val 0.853) for up to 2 more epochs, to
# sharpen the shared representation for pose rescoring (the ensemble of readout
# heads plateaued at DP@2 89.8% -- the encoder is the ceiling). Same 1.135B-token
# corpus (leak verified benign: CASF docking power equal on in/out-of-PLINDER-train
# targets). gpu_1 = full H100, ~8h/epoch @ ~2 it/s, early-stop 2 -> <=24h.
# Migrated from an interactive run before the session closed.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

.venv/bin/python pipelines/train/mlm.py \
    --token-dir data/lm_tokens_pretrain_rescore_bio \
    --atom-codebook-size 8192 \
    --micro-batch-size 256 \
    --num-workers 7 \
    --max-epochs 2 \
    --early-stop-patience 2 \
    --init-from pocket-ligand-mlm/j90rlrgm/checkpoints/mlm-e01-vl0.8528.ckpt \
    --run-name mlm_allatom_v3_cont

echo "MLM CONTINUATION DONE"

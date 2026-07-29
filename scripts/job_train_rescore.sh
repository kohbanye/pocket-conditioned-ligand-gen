#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=8:00:00
#$ -N train_rescore

# Fine-tune the pose-scoring head: warm-start the ESM3-style encoder from the
# production MLM (val 0.853, 55% zero-shot docking power) and train an MLP head
# to regress pose RMSD on the rigid-perturbation decoys. Discriminative
# complement to zero-shot PLL; target is to beat 55% docking power on CASF-2016.
# Single GPU (gpu_1). Held-out-PDB val + early stop. WANDB offline.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

.venv/bin/python pipelines/train/head.py \
    --token-dir data/lm_tokens_decoys \
    --mlm-ckpt pocket-ligand-mlm/j90rlrgm/checkpoints/mlm-e01-vl0.8528.ckpt \
    --micro-batch-size 32 \
    --num-workers 7 \
    --max-epochs 12 \
    --early-stop-patience 3 \
    --run-name rescore_head_v1

echo "RESCORE HEAD TRAIN DONE"

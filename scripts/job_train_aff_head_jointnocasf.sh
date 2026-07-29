#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=4:00:00
#$ -N jaff_head

# ABLATION joint (apples-to-apples control): affinity head on the JOINT
# leak-free MLM backbone (wxlhgqx3) + joint-tokenized Kd/Ki complexes, SAME
# protocol as the separate affinity head (mean pool, --label-cap 13). Fair
# joint-side comparison so only the tokenizer differs. ~0.1 GPU-h.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python pipelines/train/head.py \
    --token-dir data/lm_tokens_affinity_kdki \
    --mlm-ckpt pocket-ligand-mlm/wxlhgqx3/checkpoints/mlm-e02-vl0.8199.ckpt \
    --atom-codebook-size 8192 \
    --pooling mean --label-cap 13.0 \
    --micro-batch-size 32 --num-workers 8 --max-epochs 15 --early-stop-patience 3 \
    --run-name aff_head_jointnocasf

echo "JOINT-NOCASF AFFINITY HEAD DONE"

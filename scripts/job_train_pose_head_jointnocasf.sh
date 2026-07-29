#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=6:00:00
#$ -N jpose_head

# ABLATION joint (apples-to-apples control): pose head on the JOINT leak-free
# MLM backbone (wxlhgqx3) + joint-tokenized decoys, SAME protocol as the
# separate pose head. This is the fair joint-side comparison for the ablation
# (the paper's headline pose head used the interface-rich j90rlrgm backbone;
# here both sides use the nocasf backbone so only the tokenizer differs).
# Inputs already exist; no hold_jid needed. ~0.5 GPU-h.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python pipelines/train/head.py \
    --token-dir data/lm_tokens_decoys_v2 \
    --mlm-ckpt pocket-ligand-mlm/wxlhgqx3/checkpoints/mlm-e02-vl0.8199.ckpt \
    --atom-codebook-size 8192 \
    --micro-batch-size 32 --num-workers 7 --max-epochs 15 --early-stop-patience 3 \
    --run-name pose_head_jointnocasf

echo "JOINT-NOCASF POSE HEAD DONE"

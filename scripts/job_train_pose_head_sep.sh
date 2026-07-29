#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=6:00:00
#$ -N spose_head

# ABLATION separate: pose (RMSD) head on the SEPARATE MLM backbone + separate-
# tokenized decoys. Combined code space => --atom-codebook-size 16384.
# Chain: qsub -hold_jid smlm_nocasf,stok_decoysv2. ~0.5 GPU-h (head is cheap).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python pipelines/train/head.py \
    --token-dir data/lm_tokens_decoys_v2_sep \
    --mlm-ckpt pocket-ligand-mlm/mlm_nocasf_sep/checkpoints/last.ckpt \
    --atom-codebook-size 16384 \
    --micro-batch-size 32 --num-workers 7 --max-epochs 15 --early-stop-patience 3 \
    --run-name pose_head_sep

echo "SEPARATE POSE HEAD DONE"

#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=6:00:00
#$ -N spose_head4096

# FAIR-ABLATION REDO (separate 4096+4096 -> combined 8192): pose (RMSD) head on
# the SEPARATE-4096 MLM backbone + separate-4096-tokenized decoys. Combined code
# space => --atom-codebook-size 8192.
# Chain: qsub -hold_jid smlm_nocasf4096,stok_decoysv24096. ~0.5 GPU-h (head is cheap).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python pipelines/train/head.py \
    --token-dir data/lm_tokens_decoys_v2_sep4096 \
    --mlm-ckpt pocket-ligand-mlm/mlm_nocasf_sep4096/checkpoints/last.ckpt \
    --atom-codebook-size 8192 \
    --micro-batch-size 32 --num-workers 7 --max-epochs 15 --early-stop-patience 3 \
    --run-name pose_head_sep4096

echo "SEPARATE4096 POSE HEAD DONE"

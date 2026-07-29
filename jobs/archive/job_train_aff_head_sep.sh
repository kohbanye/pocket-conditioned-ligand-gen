#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=4:00:00
#$ -N saff_head

# ABLATION separate: affinity (pK) head on the SEPARATE MLM backbone +
# separate-tokenized Kd/Ki complexes. mean pool, --label-cap 13, combined
# code space => --atom-codebook-size 16384.
# Chain: qsub -hold_jid smlm_nocasf,stok_kdki. ~0.1 GPU-h.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python pipelines/train/head.py \
    --token-dir data/lm_tokens_affinity_kdki_sep \
    --mlm-ckpt pocket-ligand-mlm/mlm_nocasf_sep/checkpoints/last.ckpt \
    --atom-codebook-size 16384 \
    --pooling mean --label-cap 13.0 \
    --micro-batch-size 32 --num-workers 8 --max-epochs 15 --early-stop-patience 3 \
    --run-name aff_head_sep

echo "SEPARATE AFFINITY HEAD DONE"

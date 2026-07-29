#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=3:00:00
#$ -N arp_sep

# STAGE-3 RECOVERY job for the SEPARATE generation-LM chain (job_ar_chain_sep.sh).
# The chain's 23h h_rt cannot reach stage 3: pretrain took ~3.5h and fullft runs
# ~6.7h/epoch x 3ep, so the chain is killed mid-fullft-epoch-2 with placement
# never started. train_lm.py validates once per epoch (no val_check_interval), so
# fullft's last.ckpt is its last COMPLETED epoch -- a valid warm-start.
#
# This job runs ONLY stage 3 (placement, goodmix_sep ~286M tok x4ep, ~1-2h) and
# produces lm_placement_sep, the checkpoint ctbench/variants.py expects for the
# separate generation arm. Combined 16384-code space (--atom-codebook-size 16384).
# Submit with -hold_jid <chain job id> so it starts the moment the chain job ends
# (by qdel or walltime).
#
# Node: node_f x1. Runtime: ~1-2h (h_rt 3h).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

echo "=== AR SEP stage 3/3 (recovery): placement ==="
.venv/bin/python scripts/train_lm.py \
    --token-dir data/lm_tokens_goodmix_sep \
    --atom-codebook-size 16384 \
    --mask-prompt \
    --init-from pocket-ligand-lm/lm_fullft_sep/checkpoints/last.ckpt \
    --micro-batch-size 64 \
    --lr 5e-5 \
    --max-epochs 4 \
    --early-stop-patience 2 \
    --run-name lm_placement_sep

echo "AR SEP PLACEMENT DONE (-> lm_placement_sep)"

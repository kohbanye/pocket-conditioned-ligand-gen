#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=6:00:00
#$ -N lm_ftmix

# Condition-only fine-tune on the COMBINED complex corpus (PLINDER ~215k
# drug-like pocket-ligand complexes over ~300k distinct pockets + pocket-split
# capped CrossDocked), warm-started from the mixed-pretrain checkpoint.
# --mask-prompt trains only the <l> ligand block. Model selection uses a
# HELD-OUT-POCKET val (built by build_mixed_pretrain_cache from both corpora's
# val splits) with EarlyStopping, so we pick a GENERALISING model rather than
# the most-overfit epoch (the failure mode of the first finetune).
# DDP over 4 H100s. WANDB offline.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline

.venv/bin/python scripts/train_lm.py \
    --token-dir data/lm_tokens_finetune_mixed \
    --atom-codebook-size 8192 \
    --mask-prompt \
    --init-from pocket-ligand-lm/vwvg82y2/checkpoints/lm-e02-vl2.1796.ckpt \
    --micro-batch-size 64 \
    --max-epochs 10 \
    --lr 3e-4 \
    --early-stop-patience 2 \
    --run-name lm_finetune_mixed_v1

#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=10:00:00
#$ -N finetune_lm_cd
#$ -o finetune_lm_cd.$JOB_ID.out
#$ -e finetune_lm_cd.$JOB_ID.err
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$(pwd)"
module load cuda

# Stage 2: fine-tune the GEOM-pretrained LM on CrossDocked complexes. Warm-start
# the weights (fresh optimizer + LR schedule, NOT a Lightning resume) from the
# pretrain checkpoint (best val/loss = epoch 1 of run gdnesyzx).
# 3 epochs ~= 4.5-5 h on node_f (per the pretrain run's ~1h35m/epoch); h_rt 10 h
# leaves ~2x margin.
# Submit: qsub -g tga-ohuelab scripts/finetune_lm_crossdocked.sh
PRETRAIN_CKPT="pocket-ligand-lm/gdnesyzx/checkpoints/lm-e01-vl1.8593.ckpt"

uv run python scripts/train_lm.py \
    --token-dir data/lm_tokens \
    --run-name lm_cd_finetune \
    --max-epochs 3 \
    --init-from "$PRETRAIN_CKPT"

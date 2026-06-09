#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=24:00:00
#$ -N train_lm
#$ -o train_lm.$JOB_ID.out
#$ -e train_lm.$JOB_ID.err
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"
module load cuda

# Dense Qwen3 (~0.3B) from-scratch LM on packed VQ-VAE tokens.
# node_f = 4x H100; Lightning auto-detects the GPUs and uses DDP.
# Submit with: qsub -g <group> scripts/train_lm.sh
uv run python scripts/train_lm.py \
    --token-dir data/lm_tokens \
    --run-name lm_10ep \
    --max-epochs 10

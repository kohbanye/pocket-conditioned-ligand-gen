#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=20:00:00
#$ -N vqvae_b1a
#$ -o vqvae_b1a.$JOB_ID.out
#$ -e vqvae_b1a.$JOB_ID.err
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"
module load cuda

# Batch 1 — A: +unit-circle penalty only (shared descriptor_cache with baseline)
uv run python scripts/train_vqvae.py \
    --from-hub --hub-repo-id kohbanye/crossdocked2020 \
    --source-types cdonly it0 it2_redocked \
    --circle-loss-weight 0.1 \
    --run-name "B1-A_circle0.1"

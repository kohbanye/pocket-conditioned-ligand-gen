#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=20:00:00
#$ -N vqvae_b1b
#$ -o vqvae_b1b.$JOB_ID.out
#$ -e vqvae_b1b.$JOB_ID.err
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"
module load cuda

# Batch 1 — B: +coord_loss warmup only (shared descriptor_cache with baseline)
uv run python scripts/train_vqvae.py \
    --from-hub --hub-repo-id kohbanye/crossdocked2020 \
    --source-types cdonly it0 it2_redocked \
    --coord-loss-warmup-epochs 10 \
    --run-name "B1-B_warmup10"

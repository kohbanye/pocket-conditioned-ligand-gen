#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=20:00:00
#$ -N vqvae_fold0
#$ -o vqvae_fold0.$JOB_ID.out
#$ -e vqvae_fold0.$JOB_ID.err
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"
module load cuda

# First training run on the official-fold-split data: descriptor_cache_v3
# was built with schema v3 (per-entry pair_idx) so _setup_from_shards uses
# the manifest's cdonly/it0/it2_redocked fold0 train/test split, with val
# carved out of train at 9:1 (seed=42).
# Reuses the B2-B winner config (skip sincos norm + coord-loss warmup 10).
uv run python scripts/train_vqvae.py \
    --from-hub --hub-repo-id kohbanye/crossdocked2020 \
    --source-types cdonly it0 it2_redocked \
    --cache-dir data/descriptor_cache_v3 \
    --skip-sincos-norm \
    --coord-loss-warmup-epochs 10 \
    --run-name "fold0_skip_sincos+warmup10"

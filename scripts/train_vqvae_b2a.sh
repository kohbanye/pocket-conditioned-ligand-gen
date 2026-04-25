#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=20:00:00
#$ -N vqvae_b2a
#$ -o vqvae_b2a.$JOB_ID.out
#$ -e vqvae_b2a.$JOB_ID.err
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"
module load cuda

# Batch 2 — A: sin/cos normalization skip only (no warmup, no circle penalty).
# Uses descriptor_cache_v2/ — its normalization_stats.pt must be pre-generated
# via `scripts/recompute_norm_stats.py --cache-dir data/descriptor_cache_v2
# --skip-sincos-norm`.  --skip-sincos-norm here is redundant when the stats
# already bake in the override, but we pass it anyway for clarity.
uv run python scripts/train_vqvae.py \
    --from-hub --hub-repo-id kohbanye/crossdocked2020 \
    --source-types cdonly it0 it2_redocked \
    --cache-dir data/descriptor_cache_v2 \
    --skip-sincos-norm \
    --run-name "B2-A_skip_sincos"

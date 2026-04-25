#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=20:00:00
#$ -N vqvae_b2b
#$ -o vqvae_b2b.$JOB_ID.out
#$ -e vqvae_b2b.$JOB_ID.err
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"
module load cuda

# Batch 2 — B: sin/cos normalization skip + coord_loss warmup (B1-B winner).
# circle_loss dropped — B1-A showed it hurt 3D RMSD when sin/cos were still
# std-normalised; re-evaluate with λ=0.01 only if B2 plateaus.
uv run python scripts/train_vqvae.py \
    --from-hub --hub-repo-id kohbanye/crossdocked2020 \
    --source-types cdonly it0 it2_redocked \
    --cache-dir data/descriptor_cache_v2 \
    --skip-sincos-norm \
    --coord-loss-warmup-epochs 10 \
    --run-name "B2-B_skip_sincos+warmup10"

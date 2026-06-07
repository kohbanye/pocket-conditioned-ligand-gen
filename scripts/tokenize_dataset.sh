#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=6:00:00
#$ -N tokenize_lm
#$ -o tokenize_lm.$JOB_ID.out
#$ -e tokenize_lm.$JOB_ID.err
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"
module load cuda

CACHE=data/descriptor_cache_full
VQVAE_STATS=data/descriptor_cache_v4/normalization_stats.pt
CKPT="pocket-ligand-vqvae/3dvcbp0h/checkpoints/vqvae-epoch=99-val/ligand_coord=0.1501.ckpt"

# Stage 2: encode the full descriptor cache into LM token streams.
# CRITICAL: use the VQ-VAE's training-time normalization (v4 stats), NOT stats
# recomputed over the 25M cache -- otherwise the frozen encoder sees mismatched
# inputs and emits garbage tokens. Copying the file in also lets setup() skip
# the redundant Welford pass over the cache.
cp -n "$VQVAE_STATS" "$CACHE/normalization_stats.pt"

# Submit after Stage 1 finishes (chained):
#   qsub -g tga-ohuelab -hold_jid prep_desc_full scripts/tokenize_dataset.sh
uv run python scripts/tokenize_dataset.py \
    --from-hub --hub-repo-id kohbanye/crossdocked2020 \
    --source-types cdonly it0 it2_redocked other \
    --cache-dir "$CACHE" \
    --ckpt "$CKPT" \
    --ligand-codebook-size 4096 \
    --protein-codebook-size 8192 \
    --norm-stats "$VQVAE_STATS" \
    --out-dir data/lm_tokens \
    --batch-size 1024

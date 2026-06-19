#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=20:00:00
#$ -N vqvae_clash
#$ -o vqvae_clash.$JOB_ID.out
#$ -e vqvae_clash.$JOB_ID.err
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$(pwd)"
module load cuda

# Retrain the joint VQ-VAE with ONLY a loss change vs the 3dvcbp0h baseline: a
# clash hinge on reconstructed ligand atom pairs closer than 1.2 A (weight 5.0,
# ligand-only). Directly attacks the diagnosed failure -- the per-atom coord
# decode is pairwise-blind, so ~77% of VQ reconstructions have a sub-1.2 A clash
# (vs ~10% for GT). No new decode head (the Stage-1 knn-offset head overloaded
# the 8-D latent and made things worse; this avoids that by constraining the
# existing coord output).
#
# Reuses descriptor_cache_v4 (no regeneration); train_vqvae.py skips the
# inode-heavy ligand extraction when the cache exists. Same fold split /
# codebook sizes as 3dvcbp0h for a fair comparison.
# Submit: qsub -g tga-ohuelab scripts/train_vqvae_clash.sh
uv run python scripts/train_vqvae.py \
    --from-hub --hub-repo-id kohbanye/crossdocked2020 \
    --source-types cdonly it0 it2_redocked \
    --cache-dir data/descriptor_cache_v4 \
    --ligand-codebook-size 4096 \
    --protein-codebook-size 8192 \
    --run-name "clash_loss"

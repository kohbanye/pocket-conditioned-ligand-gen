#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=20:00:00
#$ -N vqvae_stage1
#$ -o vqvae_stage1.$JOB_ID.out
#$ -e vqvae_stage1.$JOB_ID.err
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$(pwd)"
module load cuda

# Stage 1: retrain the joint VQ-VAE with the added ligand knn_offsets DECODE
# head (LIGAND_RECON_HEADS) as a pure training-time regulariser — the code is
# pushed to encode local geometry. Decode still uses absolute coords only
# (diagnose_geometry default), so there is NO neighbour-correspondence step.
# Goal: measure how much this regularisation alone lowers the ~77% VQ-recon
# clash floor (vs the old 3dvcbp0h VQ-VAE).
#
# Uses the EXISTING descriptor_cache_v4 (knn offsets are already cached as
# input; we only added them as a decode target -> NO cache regeneration, and
# train_vqvae.py now skips the inode-heavy ligand extraction when the cache
# exists). Same fold split / codebook sizes as 3dvcbp0h for a fair comparison.
# Submit: qsub -g tga-ohuelab scripts/train_vqvae_stage1.sh
uv run python scripts/train_vqvae.py \
    --from-hub --hub-repo-id kohbanye/crossdocked2020 \
    --source-types cdonly it0 it2_redocked \
    --cache-dir data/descriptor_cache_v4 \
    --ligand-codebook-size 4096 \
    --protein-codebook-size 8192 \
    --run-name "stage1_knn_reg"

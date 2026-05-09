#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=20:00:00
#$ -N vqvae_v4
#$ -o vqvae_v4.$JOB_ID.out
#$ -e vqvae_v4.$JOB_ID.err
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"
module load cuda

# First full-data run on the spherical multi-head VQ-VAE (cache schema v4).
# Uses the cdonly fold0 manifest split, identical to the v3 baseline so
# the per-atom RMSD / Kabsch-RMSD comparison vs the Z-matrix baseline is
# apples-to-apples on the same test set.
uv run python scripts/train_vqvae.py \
    --from-hub --hub-repo-id kohbanye/crossdocked2020 \
    --source-types cdonly it0 it2_redocked \
    --cache-dir data/descriptor_cache_v4 \
    --run-name "v4_spherical_multihead"

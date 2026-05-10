#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=20:00:00
#$ -N vqvae_v4_2x
#$ -o vqvae_v4_2x.$JOB_ID.out
#$ -e vqvae_v4_2x.$JOB_ID.err
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"
module load cuda

# v4 ablation: 2x codebook (ligand 2048->4096, protein 4096->8192).
# Same descriptor cache (v4) and same fold0 split, only the model
# capacity changes, so the run is directly comparable to vqvae_v4.
uv run python scripts/train_vqvae.py \
    --from-hub --hub-repo-id kohbanye/crossdocked2020 \
    --source-types cdonly it0 it2_redocked \
    --cache-dir data/descriptor_cache_v4 \
    --ligand-codebook-size 4096 \
    --protein-codebook-size 8192 \
    --run-name "v4_codebook_2x"

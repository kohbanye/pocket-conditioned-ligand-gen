#!/bin/sh
#$ -cwd
#$ -l cpu_40=1
#$ -l h_rt=2:00:00
#$ -N prep_desc_v4
#$ -o prep_desc_v4.$JOB_ID.out
#$ -e prep_desc_v4.$JOB_ID.err
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"

# Schema v4: spherical-from-pocket-centroid descriptors with embedded
# atom-feature columns (element, charge, hybrid, aromatic, ring_size,
# numH for ligand; AA + KNN hints for protein). Earlier cache_v3 is
# still readable but uses the legacy Z-matrix layout.
uv run python scripts/prepare_descriptors.py \
    --cache-dir data/descriptor_cache_v4 \
    --num-workers 40 \
    --from-hub --hub-repo-id kohbanye/crossdocked2020 \
    --source-types cdonly it0 it2_redocked

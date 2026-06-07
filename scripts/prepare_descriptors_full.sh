#!/bin/sh
#$ -cwd
#$ -l cpu_160=1
#$ -l h_rt=10:00:00
#$ -N prep_desc_full
#$ -o prep_desc_full.$JOB_ID.out
#$ -e prep_desc_full.$JOB_ID.err
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"

# Stage 1: full 25M descriptor cache (cdonly + it0 + it2_redocked + other).
# CPU-only (RDKit/numpy); ~25M small-file reads from Lustre dominate runtime.
# Output: data/descriptor_cache_full/ (~360GB, ~502 shards).
# Submit with: qsub -g tga-ohuelab scripts/prepare_descriptors_full.sh
uv run python scripts/prepare_descriptors.py \
    --cache-dir data/descriptor_cache_full \
    --num-workers 80 \
    --from-hub --hub-repo-id kohbanye/crossdocked2020 \
    --source-types cdonly it0 it2_redocked other

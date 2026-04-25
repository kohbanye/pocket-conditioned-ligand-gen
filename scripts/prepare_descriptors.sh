#!/bin/sh
#$ -cwd
#$ -l cpu_40=1
#$ -l h_rt=24:00:00
#$ -N prep_desc_v2
#$ -o prep_desc_v2.$JOB_ID.out
#$ -e prep_desc_v2.$JOB_ID.err
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"

# CPU-only descriptor generation: no GPU module load needed.
# Writes shards to data/descriptor_cache_v2/ so the in-flight training jobs
# (reading data/descriptor_cache/) are unaffected.
uv run python scripts/prepare_descriptors.py \
    --cache-dir data/descriptor_cache_v2 \
    --num-workers 40 \
    --from-hub --hub-repo-id kohbanye/crossdocked2020 \
    --source-types cdonly it0 it2_redocked

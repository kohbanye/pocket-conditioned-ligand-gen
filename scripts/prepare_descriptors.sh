#!/bin/sh
#$ -cwd
#$ -l cpu_40=1
#$ -l h_rt=2:00:00
#$ -N prep_desc
#$ -o prep_desc.$JOB_ID.out
#$ -e prep_desc.$JOB_ID.err
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"

uv run python scripts/prepare_descriptors.py \
    --cache-dir data/descriptor_cache \
    --num-workers 40 \
    --from-hub --hub-repo-id kohbanye/crossdocked2020 \
    --source-types cdonly it0 it2_redocked

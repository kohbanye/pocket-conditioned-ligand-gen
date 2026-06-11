#!/bin/sh
#$ -cwd
#$ -l cpu_80=1
#$ -l h_rt=16:00:00
#$ -N prep_desc_full
#$ -o prep_desc_full.$JOB_ID.out
#$ -e prep_desc_full.$JOB_ID.err
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"

# Stage 1: full descriptor cache from the clean cd/it set, ALL poses per SDF
# (~36.9M poses -> ~1.4B LM tokens).
# LOW-INODE: streams ligands directly from the packed tars (no per-pose file
# extraction). Requires the snapshot (data/hub_cache/repo) + extracted
# receptors (data/hub_cache/receptors). CPU-only (RDKit/numpy). One worker per
# tar (35). Output: data/descriptor_cache_full/ shards (~520GB).
# Submit with: qsub -g tga-ohuelab scripts/prepare_descriptors_full.sh
uv run python scripts/prepare_descriptors_tar.py \
    --repo-dir data/hub_cache/repo \
    --receptors-dir data/hub_cache/receptors \
    --cache-dir data/descriptor_cache_full \
    --source-types cdonly it0 it2_redocked \
    --num-workers 35

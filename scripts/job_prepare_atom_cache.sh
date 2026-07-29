#!/bin/sh
#$ -cwd
#$ -l cpu_40=1
#$ -l h_rt=1:00:00
#$ -N atom_cache

# Build the unified all-atom descriptor cache (CPU, tar streaming, inode-safe).
# cdonly, label==1 AND *_min.sdf.gz (clean minimized near-native poses only),
# all heavy atoms of pocket residues within 8 A (<=50 residues). Writes
# data/descriptor_cache_allatom (~14 GB of shards, ~351k complexes).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"

.venv/bin/python pipelines/corpora/build_descriptors.py \
    --repo-dir data/hub_cache/repo \
    --receptors-dir data/hub_cache/receptors \
    --cache-dir data/descriptor_cache_allatom \
    --source-types cdonly \
    --max-residues 50 \
    --num-workers 35

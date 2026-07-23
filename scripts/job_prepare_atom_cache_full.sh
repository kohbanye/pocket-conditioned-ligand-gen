#!/bin/sh
#$ -cwd
#$ -l cpu_80=1
#$ -l h_rt=12:00:00
#$ -N atom_cache_full

# Stage 1 (all-atom, FULL): build the all-atom descriptor cache over the SAME
# CrossDocked pose set the 2-codebook LM used -- all 3 source types, ALL poses
# incl. decoys (--include-decoys --keep-label1-docked). ~36.7M poses, the
# all-atom counterpart of the legacy descriptor_cache_full (511G, ligand-only).
#
# Calibrated: 1 worker = 7.4 files/s (108 poses/s); 35 tars run concurrently
# (1 worker/tar) so wall-clock ~= one-tar time ~3-4 h. Output ~1.0 TB across
# ~735 shard files (inode-safe: 50k poses/shard, streamed from the 35 tars, no
# per-pose extraction). RSS ~4.2 GB/worker -> ~175 GB peak (fits cpu_80).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
set -e

.venv/bin/python scripts/prepare_descriptors_atom.py \
    --repo-dir data/hub_cache/repo \
    --receptors-dir data/hub_cache/receptors \
    --cache-dir data/descriptor_cache_atom_full \
    --source-types cdonly it0 it2_redocked \
    --include-decoys --keep-label1-docked \
    --max-residues 50 \
    --num-workers 35

echo "ATOM CACHE FULL DONE"

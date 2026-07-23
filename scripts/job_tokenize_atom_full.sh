#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -N tok_atom_full

# Stage 2 (all-atom, FULL): encode the full descriptor cache into LM token
# streams -- the all-atom counterpart of the legacy `lm_tokens` (21.65M docs).
# pocket-split gives a leak-free held-out-pocket val + CASF-2016 holdout; the
# per-pocket cap is effectively removed (100000) so ALL poses are kept, matching
# the 2-codebook data. num-rotations 1 to match the legacy token budget (legacy
# used no rotation augmentation).
#
# CRITICAL: --norm-stats is the atom VQ-VAE's TRAINING normalization
# (descriptor_cache_allatom stats), NOT stats recomputed over the full cache --
# the frozen encoder must see the inputs it was trained on or it emits garbage.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
set -e

CKPT="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"

.venv/bin/python scripts/tokenize_dataset_atom.py \
    --ckpt "$CKPT" \
    --cache-dir data/descriptor_cache_atom_full \
    --out-dir data/lm_tokens_allatom_full \
    --source-types cdonly it0 it2_redocked \
    --norm-stats data/descriptor_cache_allatom/normalization_stats.pt \
    --pocket-split \
    --max-per-pocket 100000 \
    --pocket-val-frac 0.05 \
    --casf-pdbs data/casf2016_pdbs.txt \
    --num-rotations 1 \
    --batch-size 512

echo "TOKENIZE ATOM FULL DONE"

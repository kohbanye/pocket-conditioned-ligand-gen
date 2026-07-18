#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=4:00:00
#$ -N tokc_nocasf

# Leak-free retokenize: PLINDER drug-like complexes (<p>pocket</p><l>ligand</l>)
# with the SINGLE-book atom VQ (xzkjxu9q), CASF-2016 core held out. This is the
# DOMINANT leak source (native complex poses). node_q = 48 CPU + 1 GPU.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
set -e

CKPT="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
NORM="data/descriptor_cache_allatom/normalization_stats.pt"

.venv/bin/python scripts/tokenize_plinder_protein.py \
    --complex --ckpt "$CKPT" --norm-stats "$NORM" \
    --casf-pdbs data/casf2016_pdbs.txt \
    --num-rotations 2 --num-workers 40 --batch-size 256 --mw-min 150 --mw-max 600 \
    --out-dir data/lm_tokens_complex_plinder_nocasf

echo "PLINDER COMPLEX NOCASF TOKENIZE DONE"

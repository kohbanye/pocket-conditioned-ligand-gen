#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=6:00:00
#$ -N tokp_nocasf

# Leak-free retokenize: PLINDER pockets (protein-only) with the SINGLE-book atom
# VQ (xzkjxu9q, vocab 8199), CASF-2016 core held out. node_q = 48 CPU (40 zip-
# streaming workers) + 1 GPU (VQ encode). Feeds the leak-free MLM corpus.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
set -e

CKPT="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
NORM="data/descriptor_cache_allatom/normalization_stats.pt"

.venv/bin/python pipelines/corpora/tokenize_plinder.py \
    --ckpt "$CKPT" --norm-stats "$NORM" \
    --casf-pdbs data/casf2016_pdbs.txt \
    --num-rotations 8 --num-workers 40 --batch-size 256 \
    --out-dir data/lm_tokens_protein_plinder_nocasf

echo "PLINDER PROTEIN NOCASF TOKENIZE DONE"

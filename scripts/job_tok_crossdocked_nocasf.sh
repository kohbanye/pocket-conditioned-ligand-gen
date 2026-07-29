#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=4:00:00
#$ -N tokx_nocasf

# Leak-free retokenize: CrossDocked complexes (pocket-split, cap 32/pocket, x4
# rot) with the SINGLE-book atom VQ (xzkjxu9q), CASF-2016 core held out (169/285
# CASF PDBs are in CrossDocked). Reads the local descriptor_cache_allatom shards
# + VQ encodes. node_q = 48 CPU + 1 GPU.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
set -e

CKPT="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
NORM="data/descriptor_cache_allatom/normalization_stats.pt"

.venv/bin/python pipelines/corpora/tokenize_crossdocked.py \
    --ckpt "$CKPT" --norm-stats "$NORM" \
    --source-types cdonly --pocket-split --max-per-pocket 128 --num-rotations 4 \
    --casf-pdbs data/casf2016_pdbs.txt \
    --out-dir data/lm_tokens_allatom_nocasf

echo "CROSSDOCKED NOCASF TOKENIZE DONE"

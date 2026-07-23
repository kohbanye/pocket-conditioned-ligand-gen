#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=6:00:00
#$ -N tok_aff

# Binding-affinity corpus: ~18k BioLIP crystal complexes carrying an experimental
# Kd/Ki/IC50, labelled with pK (-log10 molar), CASF-2016 core + CrossDocked
# fold0-test held out. Crystal pose only (no decoys) -- this trains the second
# head, which answers "how tightly does it bind" rather than "is this pose right".
# node_q = 48 CPU (BioLIP zip streaming + per-ligand pocket carving) + 1 GPU (VQ).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
set -e

CKPT="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
NORM="data/descriptor_cache_allatom/normalization_stats.pt"

.venv/bin/python scripts/tokenize_biolip_affinity.py \
    --ckpt "$CKPT" --norm-stats "$NORM" \
    --casf-pdbs data/casf2016_pdbs.txt \
    --pk-min 2.0 --pk-max 13.0 \
    --out-dir data/lm_tokens_affinity

echo "AFFINITY TOKENIZE DONE"

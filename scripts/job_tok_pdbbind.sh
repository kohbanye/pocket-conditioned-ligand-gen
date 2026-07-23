#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=4:00:00
#$ -N tok_pdbb

# Tokenize the PDBbind v2020 affinity corpus (refined + general, ~16.5k
# CASF-free complexes with curated pK). This is the corpus GenScore/RTMScore
# train on; it replaces the BioLIP-scraped affinity corpus whose mixed-source
# labels the diagnosis tied to the head's molecular-size shortcut.
#
# Produces the full corpus (Kd/Ki/IC50) -- IC50 is PDBbind-curated and nearly
# doubles the data, and volume was measured to help scoring power.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
set -e

VQ="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
NORM=data/descriptor_cache_allatom/normalization_stats.pt

.venv/bin/python scripts/tokenize_pdbbind_affinity.py \
    --ckpt "$VQ" --norm-stats "$NORM" \
    --affinity-types KD,KI,IC50 \
    --out-dir data/lm_tokens_affinity_pdbbind

echo "TOK PDBBIND DONE"

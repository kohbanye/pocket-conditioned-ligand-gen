#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=6:00:00
#$ -N stok_plprot4096

# FAIR-ABLATION REDO (4096+4096 -> combined 8192) of
# job_tok_plinder_protein_nocasf_sep.sh: PLINDER pockets (protein-only) encoded
# with the SEPARATE 4096 protein-VQ + 4096 ligand-VQ (unified into one
# 2*4096=8192 code space), CASF-2016 core held out. node_q = 48 CPU (40
# zip-streaming workers) + 1 GPU (VQ encode). Feeds the leak-free MLM corpus.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
set -e

.venv/bin/python scripts/tokenize_plinder_protein.py \
    --separate-protein-ckpt pocket-ligand-vqvae/protein-vqvae-4096/checkpoints/last.ckpt \
    --separate-protein-norm data/descriptor_cache_allatom/normalization_stats_protein.pt \
    --separate-ligand-ckpt pocket-ligand-vqvae/ligand-vqvae-4096/checkpoints/last.ckpt \
    --separate-ligand-norm data/descriptor_cache_allatom/normalization_stats_ligand.pt \
    --codebook-size 4096 \
    --casf-pdbs data/casf2016_pdbs.txt \
    --num-rotations 8 --num-workers 40 --batch-size 256 \
    --out-dir data/lm_tokens_protein_plinder_nocasf_sep4096

echo "PLINDER PROTEIN NOCASF SEP4096 TOKENIZE DONE"

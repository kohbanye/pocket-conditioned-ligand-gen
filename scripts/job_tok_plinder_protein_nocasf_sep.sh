#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=6:00:00
#$ -N stok_plprot

# SEPARATE-tokenizers ablation of job_tok_plinder_protein_nocasf.sh: PLINDER
# pockets (protein-only) encoded with the SEPARATE protein-VQ + ligand-VQ
# (unified into one code space) instead of the joint single-book atom VQ,
# CASF-2016 core held out. node_q = 48 CPU (40 zip-streaming workers) + 1 GPU
# (VQ encode). Feeds the leak-free MLM corpus.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
set -e

.venv/bin/python pipelines/corpora/tokenize_plinder.py \
    --separate-protein-ckpt pocket-ligand-vqvae/protein-vqvae/checkpoints/last.ckpt \
    --separate-protein-norm data/descriptor_cache_allatom/normalization_stats_protein.pt \
    --separate-ligand-ckpt pocket-ligand-vqvae/ligand-vqvae/checkpoints/last.ckpt \
    --separate-ligand-norm data/descriptor_cache_allatom/normalization_stats_ligand.pt \
    --codebook-size 8192 \
    --casf-pdbs data/casf2016_pdbs.txt \
    --num-rotations 8 --num-workers 40 --batch-size 256 \
    --out-dir data/lm_tokens_protein_plinder_nocasf_sep

echo "PLINDER PROTEIN NOCASF SEP TOKENIZE DONE"

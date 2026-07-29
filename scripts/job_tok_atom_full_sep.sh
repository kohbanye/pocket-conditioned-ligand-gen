#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -N stok_atomfull

# SEPARATE-tokenizers ablation of job_tokenize_atom_full.sh (CrossDocked FULL,
# 2.72B-token counterpart of the legacy lm_tokens). Encodes the full descriptor
# cache into LM token streams with the SEPARATE protein-VQ + ligand-VQ (unified
# into one code space) instead of the joint single-book atom VQ. pocket-split
# gives a leak-free held-out-pocket val + CASF-2016 holdout; the per-pocket cap is
# effectively removed (100000) so ALL poses are kept. num-rotations 1 to match the
# legacy token budget.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
set -e

.venv/bin/python scripts/tokenize_dataset_atom.py \
    --separate-protein-ckpt pocket-ligand-vqvae/protein-vqvae/checkpoints/last.ckpt \
    --separate-protein-norm data/descriptor_cache_allatom/normalization_stats_protein.pt \
    --separate-ligand-ckpt pocket-ligand-vqvae/ligand-vqvae/checkpoints/last.ckpt \
    --separate-ligand-norm data/descriptor_cache_allatom/normalization_stats_ligand.pt \
    --codebook-size 8192 \
    --cache-dir data/descriptor_cache_atom_full \
    --out-dir data/lm_tokens_allatom_full_sep \
    --source-types cdonly it0 it2_redocked \
    --pocket-split \
    --max-per-pocket 100000 \
    --pocket-val-frac 0.05 \
    --casf-pdbs data/casf2016_pdbs.txt \
    --num-rotations 1 \
    --batch-size 512

echo "TOKENIZE ATOM FULL SEP DONE"

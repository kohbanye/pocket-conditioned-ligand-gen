#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=4:00:00
#$ -N stok_cdnocasf

# SEPARATE-tokenizers ablation of job_tok_crossdocked_nocasf.sh: CrossDocked
# complexes (pocket-split, cap 128/pocket, x4 rot) encoded with the SEPARATE
# protein-VQ + ligand-VQ (unified into one code space) instead of the joint
# single-book atom VQ. CASF-2016 core held out. node_q = 48 CPU + 1 GPU.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
set -e

.venv/bin/python pipelines/corpora/tokenize_crossdocked.py \
    --separate-protein-ckpt pocket-ligand-vqvae/protein-vqvae/checkpoints/last.ckpt \
    --separate-protein-norm data/descriptor_cache_allatom/normalization_stats_protein.pt \
    --separate-ligand-ckpt pocket-ligand-vqvae/ligand-vqvae/checkpoints/last.ckpt \
    --separate-ligand-norm data/descriptor_cache_allatom/normalization_stats_ligand.pt \
    --codebook-size 8192 \
    --source-types cdonly --pocket-split --max-per-pocket 128 --num-rotations 4 \
    --casf-pdbs data/casf2016_pdbs.txt \
    --out-dir data/lm_tokens_allatom_nocasf_sep

echo "CROSSDOCKED NOCASF SEP TOKENIZE DONE"

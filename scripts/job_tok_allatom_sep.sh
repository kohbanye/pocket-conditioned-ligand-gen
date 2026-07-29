#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=6:00:00
#$ -N stok_allatom

# SEPARATE-tokenizers twin of the joint `data/lm_tokens_allatom` (211M, the
# CrossDocked good-pose corpus, x4 rot, default fold split, NOT CASF-held) that
# the joint p6lpk7br placement stage used inside goodmix. Built so goodmix_sep
# matches joint goodmix apples-to-apples (same complexes, separate tokenizer).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python pipelines/corpora/tokenize_crossdocked.py \
    --separate-protein-ckpt pocket-ligand-vqvae/protein-vqvae/checkpoints/last.ckpt \
    --separate-protein-norm data/descriptor_cache_allatom/normalization_stats_protein.pt \
    --separate-ligand-ckpt pocket-ligand-vqvae/ligand-vqvae/checkpoints/last.ckpt \
    --separate-ligand-norm data/descriptor_cache_allatom/normalization_stats_ligand.pt \
    --codebook-size 8192 \
    --cache-dir data/descriptor_cache_allatom \
    --out-dir data/lm_tokens_allatom_sep \
    --source-types cdonly \
    --num-rotations 4 \
    --batch-size 512 \
    --splits train val

echo "ALLATOM SEP TOKENIZE DONE"

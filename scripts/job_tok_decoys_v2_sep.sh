#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=6:00:00
#$ -N stok_decoysv2

# SEPARATE-tokenizers ablation of the decoys_v2 pose-head corpus (the inline
# tokenize step of job_train_rescore_v2.sh, extracted standalone): 12k complexes
# x 20 rigid+conformational decoys, encoded with the SEPARATE protein-VQ +
# ligand-VQ (unified into one code space) instead of the joint single-book atom
# VQ, CASF-2016 core held out. Walltime inherited from the combined parent job
# (tokenize + train); generous for the tokenize-only step.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

.venv/bin/python scripts/tokenize_decoys.py \
    --separate-protein-ckpt pocket-ligand-vqvae/protein-vqvae/checkpoints/last.ckpt \
    --separate-protein-norm data/descriptor_cache_allatom/normalization_stats_protein.pt \
    --separate-ligand-ckpt pocket-ligand-vqvae/ligand-vqvae/checkpoints/last.ckpt \
    --separate-ligand-norm data/descriptor_cache_allatom/normalization_stats_ligand.pt \
    --codebook-size 8192 \
    --casf-pdbs data/casf2016_pdbs.txt \
    --n-complexes 12000 --n-decoys 20 --out-dir data/lm_tokens_decoys_v2_sep

echo "DECOYS V2 SEP TOKENIZE DONE"

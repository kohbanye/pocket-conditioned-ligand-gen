#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=10:00:00
#$ -N lvqvae

# ABLATION: ligand-only all-atom VQ-VAE, trained on the SAME complexes as the
# joint tokenizer but on ligand atoms ONLY (--modality ligand). One codebook
# (8192), 100 epochs. Checkpoints under pocket-ligand-vqvae/ligand-vqvae/,
# own normalization_stats_ligand.pt. Ligand streams are short, so this is
# faster than the protein-only run.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline

.venv/bin/python pipelines/train/vqvae.py \
    --source-types cdonly \
    --cache-dir data/descriptor_cache_allatom \
    --codebook-size 8192 \
    --mol-batch-size 256 \
    --num-workers 8 \
    --max-epochs 100 \
    --modality ligand \
    --run-name ligand-vqvae

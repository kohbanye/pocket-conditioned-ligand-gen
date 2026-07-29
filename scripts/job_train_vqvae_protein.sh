#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=14:00:00
#$ -N pvqvae

# ABLATION: protein-only all-atom VQ-VAE, trained on the SAME complexes as the
# joint tokenizer but on protein atoms ONLY (--modality protein). One codebook
# (8192), 100 epochs. Checkpoints under pocket-ligand-vqvae/protein-vqvae/,
# own normalization_stats_protein.pt. Pairs with the ligand-only VQ to form the
# "separate tokenizers" arm vs the joint tokenizer.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline

.venv/bin/python scripts/train_vqvae_atom.py \
    --source-types cdonly \
    --cache-dir data/descriptor_cache_allatom \
    --codebook-size 8192 \
    --mol-batch-size 256 \
    --num-workers 8 \
    --max-epochs 100 \
    --modality protein \
    --run-name protein-vqvae

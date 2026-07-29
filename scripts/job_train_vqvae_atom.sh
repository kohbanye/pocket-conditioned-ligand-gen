#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=16:00:00
#$ -N atom_vqvae

# Train the unified all-atom VQ-VAE (one codebook over protein + ligand atoms)
# on data/descriptor_cache_allatom. ~7 min/epoch train + ~0.8 min/epoch val
# (GPU compute-bound, flat in batch/workers); 100 epochs ~= 13 h. Checkpoints
# top-3 by val/atom_coord (a crash still leaves usable checkpoints).
# WANDB offline so the headless job never blocks on the network (sync later).

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
    --run-name atomvqvae-v1

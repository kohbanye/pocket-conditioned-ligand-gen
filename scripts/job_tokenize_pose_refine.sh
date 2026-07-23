#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=3:00:00
#$ -N tok_refine

# Build the pose-refiner training set for the LEGACY generation decoder.
# ~12k CASF/sbdd-excluded BioLIP2 native complexes; for each we manufacture the
# exact deployment corruption -- the legacy ligand VQ-VAE (3dvcbp0h) round-trip
# of the crystal ligand (the same pairwise-blind, clash-prone reconstruction the
# LM decode emits) -- paired 1:1 with the crystal pose x1, plus 3 graded
# corruption records each (rigid + jitter). Pocket atoms (coords + chemistry)
# come from the all-atom receptor parse, stored ONCE per complex (inode-safe
# concatenated memmaps). Single GPU; WANDB not used.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python scripts/tokenize_pose_refine.py \
    --decoder legacy \
    --ckpt "pocket-ligand-vqvae/3dvcbp0h/checkpoints/vqvae-epoch=99-val/ligand_coord=0.1501.ckpt" \
    --cache-dir data/descriptor_cache_v4 \
    --casf-pdbs data/casf2016_pdbs.txt \
    --n-complexes 12000 \
    --n-corrupt 4 \
    --out-dir data/pose_refine_legacy

echo "POSE-REFINE TOKENIZE DONE"

#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=3:00:00
#$ -N tok_decoys

# Build the RMSD-labelled decoy training set for the pose-scoring head.
# 15k CASF-excluded BioLIP native complexes x (1 native + 16 rigid-perturbation
# decoys) ~= 255k poses spanning RMSD 0-8 A (protein pocket fixed, ligand
# rotated+translated by graded magnitude; RMSD known exactly). Single all-atom
# codebook (VQ xzkjxu9q) => vocab 8199. Single-process (per-pose VQ encode).
# WANDB not used.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python pipelines/corpora/tokenize_decoys.py \
    --ckpt "pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt" \
    --norm-stats data/descriptor_cache_allatom/normalization_stats.pt \
    --casf-pdbs data/casf2016_pdbs.txt \
    --n-complexes 15000 \
    --n-decoys 16 \
    --out-dir data/lm_tokens_decoys

echo "DECOY TOKENIZE DONE"

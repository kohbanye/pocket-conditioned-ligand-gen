#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=8:00:00
#$ -N refine_pipe_atom

# Pose-refiner pipeline (ALL-ATOM decoder variant), single gpu_1 job:
#   1. tokenize the pose-refine set using the unified all-atom VQ-VAE (xzkjxu9q)
#      round-trip as x0 -> data/pose_refine_atom
#   2. train the e3nn pose refiner (single-shot x1-prediction) -> refine_atom_v1
# The refiner matches the all-atom generation decoder. WANDB offline.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

.venv/bin/python pipelines/corpora/tokenize_pose_refine.py \
    --ckpt "pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt" \
    --norm-stats data/descriptor_cache_allatom/normalization_stats.pt \
    --codebook-size 8192 \
    --casf-pdbs data/casf2016_pdbs.txt \
    --n-complexes 8000 \
    --n-corrupt 4 \
    --out-dir data/pose_refine_atom

echo "TOKENIZE DONE, starting train"

.venv/bin/python pipelines/train/refiner.py \
    --data-dir data/pose_refine_atom \
    --micro-batch-size 32 \
    --num-workers 7 \
    --max-epochs 12 \
    --early-stop-patience 6 \
    --run-name refine_atom_v1

echo "REFINE PIPELINE atom DONE"

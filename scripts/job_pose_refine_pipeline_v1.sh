#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=8:00:00
#$ -N refine_pipe_v1

# Pose-refiner pipeline v1 (single gpu_1 job, one queue wait):
#   1. tokenize the LEGACY-decoder pose-refine set (8k CASF/sbdd-excluded
#      BioLIP2 complexes; legacy ligand VQ-VAE 3dvcbp0h round-trip = x0, crystal
#      pose = x1, x4 graded corruption records) -> data/pose_refine_legacy
#   2. train the e3nn pose refiner (flow-matching bridge, ~1-3M params, fp32,
#      early stop) -> pocket-ligand-refine/refine_legacy_v1/checkpoints
# WANDB offline. Val logs rmsd_refined (full ODE) vs rmsd_corrupt each epoch.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

.venv/bin/python scripts/tokenize_pose_refine.py \
    --decoder legacy \
    --ckpt "pocket-ligand-vqvae/3dvcbp0h/checkpoints/vqvae-epoch=99-val/ligand_coord=0.1501.ckpt" \
    --cache-dir data/descriptor_cache_v4 \
    --casf-pdbs data/casf2016_pdbs.txt \
    --n-complexes 8000 \
    --n-corrupt 4 \
    --out-dir data/pose_refine_legacy

echo "TOKENIZE DONE, starting train"

.venv/bin/python scripts/train_pose_refine.py \
    --data-dir data/pose_refine_legacy \
    --micro-batch-size 32 \
    --num-workers 7 \
    --max-epochs 40 \
    --early-stop-patience 6 \
    --run-name refine_legacy_v1

echo "REFINE PIPELINE v1 DONE"

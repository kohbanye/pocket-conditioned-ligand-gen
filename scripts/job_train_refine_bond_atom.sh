#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=5:00:00
#$ -N refine_bond_atom

# all-atom refiner WITH the bond-graph feature (iter1 recipe: bond_embed
# zero-init, online jitter 0.3, lambda_bond 2.0, NO angle loss). Trains on the
# EXISTING all-atom pose-refine set (bonds already stored) -> no re-tokenize.
# Goal: preserve validity (0.79 -> ~0.9) + improve PB/clash like the legacy
# bond refiner, matching the 2-codebook on/off table. WANDB offline.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

.venv/bin/python pipelines/train/refiner.py \
    --data-dir data/pose_refine_atom \
    --online-jitter-sigma 0.3 \
    --lambda-bond 2.0 \
    --micro-batch-size 16 \
    --num-workers 7 \
    --max-epochs 10 \
    --early-stop-patience 5 \
    --run-name refine_atom_bond_v1

echo "REFINE BOND ATOM TRAIN DONE"

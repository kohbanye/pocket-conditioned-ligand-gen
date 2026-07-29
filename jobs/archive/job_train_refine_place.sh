#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=6:00:00
#$ -N refine_place

# /loop iter1b: PLACEMENT-aware all-atom refiner.
#
# The current refiner (refine_atom_bond_v1) is trained on VQ round-trip error +
# per-atom jitter -- i.e. LOCAL distortion. But the LM's poses are predominantly
# MIS-PLACED in the pocket (raw Vina +8.56), which is a rigid-body error the
# refiner never saw in training. New online corruption applies a random rigid
# translation (1.5 A sigma) + rotation (15 deg sigma) about the ligand centroid
# on top of the usual jitter, so the net must slide/tilt the whole ligand back
# into the pocket. lambda_pkt raised to 3.0 because the raw Vina score is
# dominated by ligand-pocket steric overlap. Keeps the bond-graph feature
# (validity) and jitter 0.3 (PoseBusters) from the winning recipe.
# Trains on the EXISTING data/pose_refine_atom -> no re-tokenize.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

.venv/bin/python pipelines/train/refiner.py \
    --data-dir data/pose_refine_atom \
    --online-jitter-sigma 0.3 \
    --online-rigid-trans 1.2 \
    --online-rigid-rot-deg 12.0 \
    --online-rigid-prob 0.5 \
    --lambda-bond 2.0 \
    --lambda-pkt 3.0 \
    --micro-batch-size 16 \
    --num-workers 7 \
    --max-epochs 12 \
    --early-stop-patience 4 \
    --run-name refine_atom_place_v2

echo "REFINE PLACE TRAIN DONE"

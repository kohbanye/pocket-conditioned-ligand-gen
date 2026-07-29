#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=5:00:00
#$ -N refine_bond

# Iteration to lift PoseBusters validity WITHOUT changing the tokenizer:
# retrain the refiner with (1) the bond-graph edge feature (net now knows which
# close pairs are bonds vs clashes) and (2) online intramolecular jitter (x0 gets
# bad bond lengths/angles the refiner must repair). Trains on the EXISTING
# legacy pose-refine set (bonds already stored) -> no re-tokenize. Single-shot
# val (n_flow_steps=1). Target: PB-valid toward the baseline band (~0.5) while
# keeping Vina Score in the band. WANDB offline.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

.venv/bin/python pipelines/train/refiner.py \
    --data-dir data/pose_refine_legacy \
    --online-jitter-sigma 0.5 \
    --lambda-bond 2.0 \
    --lambda-angle 1.0 \
    --micro-batch-size 16 \
    --num-workers 7 \
    --max-epochs 10 \
    --early-stop-patience 5 \
    --run-name refine_legacy_bond_v2

echo "REFINE BOND TRAIN DONE"

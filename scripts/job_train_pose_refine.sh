#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=6:00:00
#$ -N train_refine

# Train the E(3)-equivariant ligand pose refiner (flow-matching bridge) on the
# legacy-decoder pose-refine set. Learns to map the VQ-VAE-corrupted pose x0
# onto the crystal pose x1 conditioned on the frozen pocket -- a fast,
# differentiable, generation-time replacement for Vina local minimisation.
# Target: close the Score->Min gap (raw-pose Vina +1.21 -> toward -6.45) and
# raise PoseBusters/clash-free rates without regressing Vina min/dock.
# Small e3nn net (~1-3M params); single GPU; fp32 (e3nn TPs unstable in bf16).
# WANDB offline. Val logs rmsd_refined (full ODE) vs rmsd_corrupt each epoch.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

.venv/bin/python scripts/train_pose_refine.py \
    --data-dir data/pose_refine_legacy \
    --micro-batch-size 32 \
    --num-workers 7 \
    --max-epochs 40 \
    --early-stop-patience 5 \
    --run-name refine_legacy_v1

echo "POSE-REFINE TRAIN DONE"

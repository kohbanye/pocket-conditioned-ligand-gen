#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=4:00:00
#$ -N head_aff

# Train the AFFINITY head: same encoder + pooling as the pose head, but the MLP
# regresses pK (-log10 molar) on ~18k BioLIP crystal complexes instead of pose
# RMSD. --label-cap 13 keeps real binders from being clipped (the pose head's
# default cap of 8 is an RMSD ceiling and would destroy the pK range).
# This is what gives the model CASF scoring/ranking power, which the pose head
# structurally cannot have (measured R = -0.04).
# Warm-started from the pose backbone (j90rlrgm); CASF core is absent from the
# affinity labels, and the leaked/clean split is checked at eval time.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python pipelines/train/head.py \
    --token-dir data/lm_tokens_affinity \
    --mlm-ckpt pocket-ligand-mlm/j90rlrgm/checkpoints/mlm-e01-vl0.8528.ckpt \
    --run-name rescore_affinity \
    --max-epochs 15 --micro-batch-size 32 --early-stop-patience 3 \
    --pooling mean --label-cap 13.0 --num-workers 8

echo "AFFINITY HEAD DONE"

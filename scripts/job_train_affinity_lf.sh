#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=4:00:00
#$ -N head_afflf

# Train the AFFINITY head: same encoder + pooling as the pose head, but the MLP
# regresses pK (-log10 molar) on ~18k BioLIP crystal complexes instead of pose
# RMSD. --label-cap 13 keeps real binders from being clipped (the pose head's
# default cap of 8 is an RMSD ceiling and would destroy the pK range).
# This is what gives the model CASF scoring/ranking power, which the pose head
# structurally cannot have (measured R = -0.04).
# Warm-started from the LEAK-FREE backbone (wxlhgqx3): CASF is absent from BOTH the
# pretraining corpus and the affinity labels, so all 285 targets are truly unseen.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python scripts/train_rescore.py \
    --token-dir data/lm_tokens_affinity \
    --mlm-ckpt pocket-ligand-mlm/wxlhgqx3/checkpoints/mlm-e02-vl0.8199.ckpt \
    --run-name rescore_affinity_lf \
    --max-epochs 15 --micro-batch-size 32 --early-stop-patience 3 \
    --pooling mean --label-cap 13.0 --num-workers 8

echo "AFFINITY HEAD DONE"

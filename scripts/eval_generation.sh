#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=2:00:00
#$ -N eval_gen
#$ -o eval_gen.$JOB_ID.out
#$ -e eval_gen.$JOB_ID.err
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"
module load cuda

# Generate 100 ligands for each of 100 held-out-test pockets with the best
# 10-epoch LM, decode to 3D, and dump geometry-quality metrics for the
# notebooks/generation_eval.py notebook. ~10k molecules; run on a GPU node
# (the login node's limits kill a batch this size).
# Submit with: qsub -g tga-ohuelab scripts/eval_generation.sh
uv run python scripts/eval_generation.py \
    --lm-ckpt "pocket-ligand-lm/g79let5b/checkpoints/lm-e09-vl1.4088.ckpt" \
    --num-pockets 100 --num-samples 100 --seed 0

#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -N sbdd_gen_base

# Generate 100 ligands/pocket for ONE prior-work baseline on the CrossDocked
# 100-pocket test set, for the generation comparison table. run_generation.py
# drives the model's adapter, which subprocesses that model's own conda env
# (envs/<model>/bin/python) with its weights (weights/<model>/). Node: gpu_1.
# Set MODEL to diffsbdd | targetdiff | diffgui. Submit with -p -3 + a distinct -N.

cd /gs/bs/tga-ohuelab/sakano/git/sbdd-bench
export WANDB_MODE=offline
set -e

MODEL="${MODEL:-diffsbdd}"
.venv/bin/python scripts/run_generation.py \
    --models "$MODEL" \
    --index data/targets/index.json \
    --n-samples 100
echo "BASELINE GEN DONE ($MODEL)"

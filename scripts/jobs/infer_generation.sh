#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=6:00:00
#$ -N ctbench_generation

# Generation inference for one tokenizer variant: generate 3D ligands with the
# source repo's all-atom generator, then score with the sbdd-bench harness.
# Node: gpu_1 (1 GPU). Runtime: ~3-6 h (docking dominates). Set VARIANT to
# joint | protein_only | ligand_only. Prereq: `uv sync` on a login node first,
# plus the sbdd-bench micromamba env for scoring. See the first-run caveat in
# ctbench/inference/generation.py.

cd /gs/bs/tga-ohuelab/sakano/git/complex-tokenizer-bench
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/complex-tokenizer-bench:/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export CTBENCH_SOURCE_REPO=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export WANDB_MODE=offline
set -e

VARIANT="${VARIANT:-joint}"
.venv/bin/python scripts/infer_generation.py --variant "$VARIANT"
echo "GENERATION INFERENCE DONE ($VARIANT)"

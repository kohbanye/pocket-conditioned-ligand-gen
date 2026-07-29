#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=2:00:00
#$ -N ctbench_affinity

# Affinity inference for one tokenizer variant on CASF-2016 (285 crystal
# complexes), writing results/affinity/<variant>/<head>.csv (one per ensemble
# head). Node: gpu_1 (1 GPU). Runtime: ~1-2 h. Set VARIANT to
# joint | protein_only | ligand_only. Prereq: `uv sync` on a login node first.

cd /gs/bs/tga-ohuelab/sakano/git/complex-tokenizer-bench
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/complex-tokenizer-bench:/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export CTBENCH_SOURCE_REPO=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export WANDB_MODE=offline
set -e

VARIANT="${VARIANT:-joint}"
.venv/bin/python scripts/infer_affinity.py --variant "$VARIANT"
echo "AFFINITY INFERENCE DONE ($VARIANT)"

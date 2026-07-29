#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=4:00:00
#$ -N ctbench_rescore

# Pose-rescoring inference for one tokenizer variant on CASF-2016 (285 targets),
# writing results/rescoring/<variant>/<head>.csv. Node: gpu_1 (1 GPU). Runtime:
# ~2-4 h for the full decoy set. Set VARIANT to joint | protein_only | ligand_only.
# Prereq: run `uv sync` on a login node first (installs this repo + the source
# repo editable), so .venv exists.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/ctbench
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/ctbench:/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export CTBENCH_SOURCE_REPO=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export WANDB_MODE=offline
set -e

VARIANT="${VARIANT:-joint}"
/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/.venv/bin/python scripts/infer_rescoring.py --variant "$VARIANT"
echo "RESCORING INFERENCE DONE ($VARIANT)"

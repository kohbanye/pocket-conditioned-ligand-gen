#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=12:00:00
#$ -N ctbench_gen_cd

# Generate ligands for ONE variant on the CrossDocked2020 100-pocket test set
# (generation only; scoring is a separate CPU job -> eval_crossdocked.sh).
#
# Node: gpu_1 (1 GPU, 8 CPU, 96 GB). Runtime: ~3-6 h for 100 pockets x 100
#   samples (refiner on); h_rt 12 h is head-room. Cost ~0.2 coeff * wall.
# Set VARIANT to joint_nocasf | separate_4096 (or joint | separate).
# Prereq: `uv sync` in ctbench + sbdd-bench + source repo done on a login node,
#   and the 100-pocket index built (scripts/prepare_targets.py). SEPARATE_4096
#   also needs lm_placement_sep4096/last.ckpt to exist.
#
# Submit (state purpose/node/time, then confirm before running):
#   qsub -g <group> -v VARIANT=joint_nocasf scripts/jobs/gen_crossdocked.sh
#   qsub -g <group> -v VARIANT=separate_4096 scripts/jobs/gen_crossdocked.sh

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/ctbench
export CTBENCH_SOURCE_REPO=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export CTBENCH_SBDD_PYTHON=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/sbddbench/.venv/bin/python
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/ctbench:/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

VARIANT="${VARIANT:-separate_4096}"
/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/.venv/bin/python scripts/infer_generation_crossdocked.py \
    --variant "$VARIANT" --n-samples 100 --skip-eval
echo "CROSSDOCKED GENERATION DONE ($VARIANT)"

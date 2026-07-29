#!/bin/sh
#$ -cwd
#$ -l cpu_160=1
#$ -l h_rt=6:00:00
#$ -N ctbench_eval_cd

# Score ONE variant's 100-pocket generations with the sbdd-bench harness
# (RDKit chem + PoseBusters + AutoDock Vina Score/Min/Dock, composite hit-rate).
# Docking parallelises across all cores (multiprocessing.Pool).
#
# Node: cpu_160 (160 CPU, 368 GB, no GPU). Runtime: ~2-4 h per variant for
#   100 pockets x ~100 dockable mols x {score,min,dock} at exhaustiveness 8;
#   h_rt 6 h is head-room. Cost ~0.6 coeff * wall.
# Set VARIANT to the same value used in gen_crossdocked.sh. Writes
#   results/generation/<VARIANT>/{per_model,per_target}.csv + per_molecule.parquet.
#
# Submit (state purpose/node/time, then confirm before running):
#   qsub -g <group> -v VARIANT=joint_nocasf scripts/jobs/eval_crossdocked.sh
#   qsub -g <group> -v VARIANT=separate_4096 scripts/jobs/eval_crossdocked.sh

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/ctbench
export CTBENCH_SOURCE_REPO=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export CTBENCH_SBDD_PYTHON=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/sbddbench/.venv/bin/python
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/ctbench:/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export T4TMPDIR="${T4TMPDIR:-$HOME/tmpdir}"
set -e

VARIANT="${VARIANT:-separate_4096}"
/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/.venv/bin/python scripts/infer_generation_crossdocked.py \
    --variant "$VARIANT" --skip-gen --dock-modes score min dock
echo "CROSSDOCKED EVAL DONE ($VARIANT)"

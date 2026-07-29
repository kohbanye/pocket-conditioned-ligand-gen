#!/bin/sh
#$ -cwd
#$ -l cpu_160=1
#$ -l h_rt=6:00:00
#$ -N ctb_eval_arm

# Score ONE already-built generation arm (a directory of per-target SDFs under
# sbdd-bench/outputs/<ARM>/own/) with the sbdd-bench harness: RDKit chem +
# PoseBusters + AutoDock Vina score/min/dock + composite hit-rate. Unlike
# eval_crossdocked.sh this does NOT go through the variant registry, so it can
# score arms produced by scripts/build_constrained_arm.py.
#
# Node: cpu_160 (160 CPU, 368 GB, no GPU). Runtime ~2-4 h for 100 pockets x 100
#   molecules x {score,min,dock} at exhaustiveness 8; h_rt 6 h is head-room.
#
#   qsub -g tga-ohuelab -p -3 -v ARM=joint_bo scripts/jobs/eval_arm.sh

cd /gs/bs/tga-ohuelab/sakano/git/complex-tokenizer-bench
export T4TMPDIR="${T4TMPDIR:-$HOME/tmpdir}"
export TMPDIR="$T4TMPDIR"
mkdir -p "$TMPDIR"
set -e

ARM="${ARM:?set ARM=<output dir name under sbdd-bench/outputs>}"
SB=/gs/bs/tga-ohuelab/sakano/git/sbdd-bench
RESULTS=/gs/bs/tga-ohuelab/sakano/git/complex-tokenizer-bench/results/generation/$ARM
mkdir -p "$RESULTS"

cd "$SB"
.venv/bin/python scripts/run_evaluation.py \
    --models own \
    --index "$SB/data/targets/index.json" \
    --out-dir "$SB/outputs/$ARM" \
    --results "$RESULTS" \
    --dock-modes score min dock \
    --dock-workers 150

echo "ARM EVAL DONE ($ARM) -> $RESULTS"

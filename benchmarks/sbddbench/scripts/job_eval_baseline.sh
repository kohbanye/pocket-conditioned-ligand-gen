#!/bin/sh
#$ -cwd
#$ -l cpu_160=1
#$ -l h_rt=8:00:00
#$ -N sbdd_eval_base

# Score ONE baseline's generated SDFs on the CrossDocked 100-pocket test set with
# the shared harness (Vina score/min/dock + chem/pose/diversity). Writes into
# sbdd-bench results/; ctbench collect_baselines.py then folds it into
# results/generation/baselines/. Set MODEL to diffsbdd | targetdiff | diffgui.
cd /gs/bs/tga-ohuelab/sakano/git/sbdd-bench
export WANDB_MODE=offline
set -e
MODEL="${MODEL:-diffsbdd}"
RESULTS="${RESULTS:-results}"
# Optional IDS_FILE (path to a file of whitespace-separated target ids) restricts
# the eval to a chunk, so a heavy 100-pocket dock can be split across parallel
# cpu_160 nodes (each writes its own RESULTS dir; merge afterwards). A file path
# is used instead of inline ids so the -v value carries no spaces. Unset = full set.
IDS_ARG=""
[ -n "${IDS_FILE:-}" ] && IDS_ARG="--ids $(cat "$IDS_FILE")"
.venv/bin/python scripts/run_evaluation.py \
    --models "$MODEL" \
    --index data/targets/index.json \
    --results "$RESULTS" \
    $IDS_ARG \
    --dock-modes score min dock
echo "BASELINE EVAL DONE ($MODEL -> $RESULTS)"

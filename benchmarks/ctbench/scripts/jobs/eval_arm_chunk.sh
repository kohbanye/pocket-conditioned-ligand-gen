#!/bin/sh
#$ -cwd
#$ -l cpu_40=1
#$ -l h_rt=3:00:00
#$ -N ctb_eval_chunk

# Score one already-built generation arm, SPLIT over target chunks as an array
# job. Same total core-hours as a single cpu_160 job (4 x cpu_40 = 0.6 coeff, the
# cpu_160 coefficient) but 40-slot tasks schedule far sooner than one 160-slot
# reservation, and the chunks dock in parallel.
#
# Per-chunk results land in results/generation/<ARM>/chunk<TASK>/; merge with
# scripts/merge_eval_chunks.py (per_molecule.parquet concat) before comparing.
#
# Node: cpu_40 (40 CPU, 92 GB, no GPU) x NCHUNK tasks. Runtime ~0.5-1.5 h per
#   chunk of 25 pockets x 100 molecules x {score,min,dock}; h_rt 3 h head-room.
#
#   qsub -g tga-ohuelab -p -3 -t 1-4 -v ARM=joint_bo,NCHUNK=4 \
#        scripts/jobs/eval_arm_chunk.sh

cd /gs/bs/tga-ohuelab/sakano/git/complex-tokenizer-bench
export T4TMPDIR="${T4TMPDIR:-$HOME/tmpdir}"
export TMPDIR="$T4TMPDIR"
mkdir -p "$TMPDIR"
set -e

ARM="${ARM:?set ARM=<output dir name under sbdd-bench/outputs>}"
NCHUNK="${NCHUNK:-4}"
TASK="${SGE_TASK_ID:-1}"
SB=/gs/bs/tga-ohuelab/sakano/git/sbdd-bench
RESULTS=/gs/bs/tga-ohuelab/sakano/git/complex-tokenizer-bench/results/generation/$ARM/chunk$TASK
mkdir -p "$RESULTS"

IDS=$(awk -v t="$TASK" -v n="$NCHUNK" 'NR % n == (t % n)' data/target_ids.txt | tr '\n' ' ')
echo "eval chunk $TASK/$NCHUNK arm=$ARM targets: $IDS"

cd "$SB"
# shellcheck disable=SC2086
.venv/bin/python scripts/run_evaluation.py \
    --models own \
    --index "$SB/data/targets/index.json" \
    --out-dir "$SB/outputs/$ARM" \
    --results "$RESULTS" \
    --ids $IDS \
    --dock-modes score min dock \
    --dock-workers 38

echo "ARM EVAL CHUNK DONE ($ARM task $TASK) -> $RESULTS"

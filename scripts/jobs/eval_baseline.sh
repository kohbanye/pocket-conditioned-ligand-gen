#!/bin/sh
#$ -cwd
#$ -l cpu_40=1
#$ -l h_rt=4:00:00
#$ -N ctb_eval_base

# Score an existing baseline's generations on the 100-pocket set. targetdiff was
# generated but never evaluated, so the only baseline in results/ is DiffSBDD --
# one point of comparison is thin for a fairness claim.
#
#   qsub -g tga-ohuelab -p -3 -v MODEL=targetdiff scripts/jobs/eval_baseline.sh

cd /gs/bs/tga-ohuelab/sakano/git/sbdd-bench
export T4TMPDIR="${T4TMPDIR:-$HOME/tmpdir}"
export TMPDIR="$T4TMPDIR"
mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
set -e
MODEL="${MODEL:?set MODEL}"
RES=/gs/bs/tga-ohuelab/sakano/git/complex-tokenizer-bench/results/generation/baseline_$MODEL
mkdir -p "$RES"
.venv/bin/python scripts/run_evaluation.py --models "$MODEL" \
    --index data/targets/index.json --out-dir outputs \
    --results "$RES" --dock-modes score min --dock-workers 38
echo "BASELINE EVAL DONE ($MODEL)"

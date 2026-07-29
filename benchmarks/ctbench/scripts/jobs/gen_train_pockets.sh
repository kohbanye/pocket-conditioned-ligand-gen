#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=3:00:00
#$ -N ctb_gen_train

# Generate ligands for CrossDocked *train* pockets — the distillation corpus for
# the pose refiner. These pockets are disjoint from the 100-pocket evaluation set
# (prepare_train_pockets.py asserts it), so a refiner trained on them can be
# scored on the benchmark without leakage.
#
# Node: gpu_1 x NCHUNK array tasks. Runtime ~20 pockets x 100 samples ~ 20 min.
#
#   qsub -g tga-ohuelab -p -3 -t 1-4 -v NCHUNK=4 scripts/jobs/gen_train_pockets.sh

cd /gs/bs/tga-ohuelab/sakano/git/complex-tokenizer-bench
export CTBENCH_SOURCE_REPO=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export CTBENCH_SBDD_PYTHON=/gs/bs/tga-ohuelab/sakano/git/sbdd-bench/.venv/bin/python
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/complex-tokenizer-bench:/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export T4TMPDIR="${T4TMPDIR:-$HOME/tmpdir}"
export TMPDIR="$T4TMPDIR"
mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
set -e

VARIANT="${VARIANT:-separate_4096}"
NCHUNK="${NCHUNK:-4}"
NSAMPLES="${NSAMPLES:-100}"
TASK="${SGE_TASK_ID:-1}"
SB=/gs/bs/tga-ohuelab/sakano/git/sbdd-bench
IDX="$SB/data/train_pockets/index.json"

IDS=$(awk -v t="$TASK" -v n="$NCHUNK" 'NR % n == (t % n)' "$SB/data/train_pockets/target_ids.txt" | tr '\n' ' ')
echo "train-pocket gen task $TASK/$NCHUNK variant=$VARIANT n=$NSAMPLES targets: $IDS"

# shellcheck disable=SC2086
.venv/bin/python scripts/infer_generation_crossdocked.py \
    --variant "$VARIANT" --out-suffix _trainpk --n-samples "$NSAMPLES" \
    --skip-eval --index "$IDX" --ids $IDS

echo "TRAIN POCKET GEN DONE (task $TASK)"

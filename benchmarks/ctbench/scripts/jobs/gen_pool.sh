#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=5:00:00
#$ -N ctb_gen_pool

# Generate an OVERSAMPLED pool for one variant, as an array job over target
# chunks. The pool feeds scripts/build_constrained_arm.py, which applies the
# constrained-sampling protocol (bond-order perception + single-fragment +
# size floor tied to the reference ligand) and keeps the first 100 acceptances
# per target. Measured acceptance rate is ~0.57, so ~400 samples/target fills
# most targets; under-filled targets are topped up by the builder.
#
# Node: gpu_1 (1 GPU, 8 CPU, 96 GB) x NCHUNK tasks. Runtime: 400 samples x
#   ~20 targets/task, measured ~50 s per 100 samples per target -> ~1-2.5 h;
#   h_rt 5 h is head-room. Cost ~0.2 coeff * wall * premium priority.
#
#   qsub -g tga-ohuelab -p -3 -t 1-5 -v VARIANT=joint_nocasf,NCHUNK=5 \
#        scripts/jobs/gen_pool.sh

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/ctbench
export CTBENCH_SOURCE_REPO=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export CTBENCH_SBDD_PYTHON=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/sbddbench/.venv/bin/python
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/ctbench:/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export T4TMPDIR="${T4TMPDIR:-$HOME/tmpdir}"
export TMPDIR="$T4TMPDIR"
mkdir -p "$TMPDIR"
set -e

VARIANT="${VARIANT:?set VARIANT}"
NCHUNK="${NCHUNK:-5}"
NSAMPLES="${NSAMPLES:-400}"
TASK="${SGE_TASK_ID:-1}"
# Pool identity. The generator seeds torch from --seed, so an EXTENSION pool
# must use both a fresh suffix (its own output dir) and a fresh seed, or it
# reproduces the first pool exactly. build_constrained_arm.py concatenates
# several --pool-dir values.
SUFFIX="${SUFFIX:-_pool}"
export SBDD_OWN_SEED="${SEED:-0}"
# Reference-conditioned minimum ligand length (0 = off, 1.0 = at least as many
# heavy atoms as the target's crystal ligand) and sampling temperature.
export SBDD_OWN_MIN_ATOMS_FRAC="${MIN_ATOMS_FRAC:-0}"
export SBDD_OWN_MIN_ATOMS_ABS="${MIN_ATOMS_ABS:-0}"
if [ -n "${TEMPERATURE:-}" ]; then export SBDD_OWN_TEMPERATURE="$TEMPERATURE"; fi
if [ -n "${TOP_P:-}" ]; then export SBDD_OWN_TOP_P="$TOP_P"; fi

# Round-robin the target list so every task gets a mix of cheap and expensive
# pockets (per-target cost varies ~5x with pocket size).
IDS=$(awk -v t="$TASK" -v n="$NCHUNK" 'NR % n == (t % n)' data/target_ids.txt | tr '\n' ' ')
echo "task $TASK/$NCHUNK variant=$VARIANT n_samples=$NSAMPLES suffix=$SUFFIX seed=$SBDD_OWN_SEED min_atoms_frac=$SBDD_OWN_MIN_ATOMS_FRAC min_atoms_abs=$SBDD_OWN_MIN_ATOMS_ABS temp=${SBDD_OWN_TEMPERATURE:-default} targets: $IDS"

# shellcheck disable=SC2086
/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/.venv/bin/python scripts/infer_generation_crossdocked.py \
    --variant "$VARIANT" \
    --out-suffix "$SUFFIX" \
    --n-samples "$NSAMPLES" \
    --skip-eval \
    --ids $IDS

echo "POOL GEN DONE ($VARIANT task $TASK)"

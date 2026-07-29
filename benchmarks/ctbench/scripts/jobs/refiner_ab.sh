#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=2:00:00
#$ -N ctb_ref_ab

# A/B one pose refiner (and optionally one sampling temperature) on a target
# subset, scored with vina_score ONLY.
#
# Why this and not the vina_dock pipeline: vina_dock re-docks the molecule from a
# bare XYZ, so it ignores our coordinates and our SDF bond block entirely -- the
# refiner cannot move it (measured: identical means to 3 decimals). vina_score is
# scored on the pose AS GENERATED, so it is the metric the refiner actually
# drives. Scoring is score-only, which is ~20x cheaper than docking, so a refiner
# A/B round is minutes rather than hours.
#
# Node: gpu_1 (1 GPU, 8 CPU, 96 GB). Runtime: ~25 targets x 100 samples generation
#   (~15 min) + score-only evaluation on 7 workers (~10 min); h_rt 2 h is
#   head-room. Cost ~0.2 coeff * wall.
#
#   qsub -g tga-ohuelab -p -3 \
#     -v VARIANT=separate_4096,TAG=geo1,REFINER=pocket-ligand-refine/refine_atom_geo_v1/checkpoints/refine-e11-r0.9280.ckpt \
#     scripts/jobs/refiner_ab.sh

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/ctbench
export CTBENCH_SOURCE_REPO=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export CTBENCH_SBDD_PYTHON=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/sbddbench/.venv/bin/python
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/ctbench:/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export T4TMPDIR="${T4TMPDIR:-$HOME/tmpdir}"
export TMPDIR="$T4TMPDIR"
mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
set -e

VARIANT="${VARIANT:?set VARIANT}"
TAG="${TAG:?set TAG}"
REFINER="${REFINER:-}"
NSAMPLES="${NSAMPLES:-100}"
IDFILE="${IDFILE:-data/target_ids_ab.txt}"
if [ -n "${TEMPERATURE:-}" ]; then export SBDD_OWN_TEMPERATURE="$TEMPERATURE"; fi
if [ -n "${MIN_ATOMS_FRAC:-}" ]; then export SBDD_OWN_MIN_ATOMS_FRAC="$MIN_ATOMS_FRAC"; fi
if [ -n "${MIN_ATOMS_ABS:-}" ]; then export SBDD_OWN_MIN_ATOMS_ABS="$MIN_ATOMS_ABS"; fi

IDS=$(tr '\n' ' ' < "$IDFILE")
SUF="_ab_$TAG"
echo "refiner A/B: variant=$VARIANT tag=$TAG refiner=$REFINER temp=${SBDD_OWN_TEMPERATURE:-default} n=$NSAMPLES"

REFARG=""
[ -n "$REFINER" ] && REFARG="--refiner $REFINER"

# shellcheck disable=SC2086
/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/.venv/bin/python scripts/infer_generation_crossdocked.py \
    --variant "$VARIANT" --out-suffix "$SUF" --n-samples "$NSAMPLES" \
    --skip-eval $REFARG --ids $IDS

SB=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/sbddbench
RES=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/ctbench/results/generation/${VARIANT}${SUF}
mkdir -p "$RES"
cd "$SB"
# shellcheck disable=SC2086
/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/.venv/bin/python scripts/run_evaluation.py --models own \
    --index "$SB/data/targets/index.json" \
    --out-dir "$SB/outputs/${VARIANT}${SUF}" --results "$RES" \
    --ids $IDS --dock-modes score --dock-workers 7

echo "REFINER AB DONE ($VARIANT $TAG) -> $RES"

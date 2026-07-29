#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=3:00:00
#$ -N ctb_mlref

# Apply a TRAINED pose refiner to an arm's poses and score them with vina_score.
# This is the ML-only counterpart of scripts/jobs/relax_eval.sh: same arm, same
# input poses, no physics at inference -- the comparison a fair as-is Vina number
# against DiffSBDD/DiffGui needs.
#
# Node: gpu_1 (1 GPU, 8 CPU). Runtime ~20 min refinement + ~30 min scoring for a
#   25-target subset; h_rt 3 h covers the full 100.
#
#   qsub -g tga-ohuelab -p -3 \
#     -v ARM=sep4096_fin,OUT=fin_mlref,CKPT=pocket-ligand-refine/refine_dist_v2/checkpoints/x.ckpt,PROJECT=1 \
#     scripts/jobs/mlrefine_eval.sh

cd /gs/bs/tga-ohuelab/sakano/git/complex-tokenizer-bench
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/complex-tokenizer-bench:/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export T4TMPDIR="${T4TMPDIR:-$HOME/tmpdir}"
export TMPDIR="$T4TMPDIR"
mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export WANDB_MODE=offline
set -e

ARM="${ARM:?set ARM}"
OUT="${OUT:?set OUT}"
CKPT="${CKPT:?set CKPT}"
IDFILE="${IDFILE:-data/target_ids_ab.txt}"
SB=/gs/bs/tga-ohuelab/sakano/git/sbdd-bench
SRC=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
REPEAT="${REPEAT:-1}"
NSTEPS="${NSTEPS:-1}"
PROJARG=""
[ -n "${PROJECT:-}" ] && PROJARG="--project"

echo "ml-refine $ARM -> $OUT ckpt=$CKPT project=${PROJECT:-0}"
rm -rf "$SB/outputs/$OUT"
# shellcheck disable=SC2046
"$SRC/.venv/bin/python" scripts/apply_refiner_to_arm.py \
    --arm "$ARM" --out-arm "$OUT" --ckpt "$CKPT" $PROJARG \
    --repeat "$REPEAT" --n-steps "$NSTEPS" \
    --targets $(tr '\n' ' ' < "$IDFILE")

if [ -n "${SKIP_EVAL:-}" ]; then echo "APPLY ONLY DONE ($OUT)"; exit 0; fi

RES=/gs/bs/tga-ohuelab/sakano/git/complex-tokenizer-bench/results/generation/$OUT
mkdir -p "$RES"
cd "$SB"
# shellcheck disable=SC2046
.venv/bin/python scripts/run_evaluation.py --models own \
    --index "$SB/data/targets/index.json" \
    --out-dir "$SB/outputs/$OUT" --results "$RES" \
    --ids $(tr '\n' ' ' < /gs/bs/tga-ohuelab/sakano/git/complex-tokenizer-bench/"$IDFILE") \
    --dock-modes score --dock-workers 7

echo "ML REFINE EVAL DONE ($OUT)"

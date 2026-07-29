#!/bin/sh
#$ -cwd
#$ -l cpu_16=1
#$ -l h_rt=2:00:00
#$ -N ctb_pipe_build

# Stage 1 of the unattended refiner pipeline: turn the relaxed CrossDocked *train*
# pockets into a distillation set. Submit with -hold_jid on the relaxation jobs so
# it only starts once every teacher pose exists.
#
# The displacement filter drops pairs where the teacher ran away — on pockets taken
# straight from hub_cache the reference ligand is sometimes incomplete, pocket
# extraction then misses atoms and the relaxation diverges (mean displacement 1.43 A
# with a 19.9 A tail, against 0.52 A on clean pockets).
#
#   qsub -g tga-ohuelab -p -3 -hold_jid <relax_ids> scripts/jobs/pipe_build.sh

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/ctbench
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export T4TMPDIR="${T4TMPDIR:-$HOME/tmpdir}"; export TMPDIR="$T4TMPDIR"; mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
set -e

SRC=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
SB=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/sbddbench
OUT="${OUT:-$SRC/data/pose_refine_tp400}"

echo "relaxed pockets available: $(find $SB/outputs/T_trainpk/own -name generated.sdf | wc -l)"
rm -rf "$OUT"
"$SRC/.venv/bin/python" scripts/build_distill_refine_set.py \
    --pairs separate_4096_trainpk:T_trainpk \
    --index "$SB/data/train_pockets/index.json" \
    --out-dir "$OUT" \
    --val-targets 20 --max-per-target 100 --max-disp 2.5

echo "PIPE BUILD DONE -> $OUT"

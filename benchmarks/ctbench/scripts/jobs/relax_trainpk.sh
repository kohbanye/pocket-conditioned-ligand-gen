#!/bin/sh
#$ -cwd
#$ -l cpu_40=1
#$ -l h_rt=2:00:00
#$ -N ctb_relax_tp

# Ideal-geometry teacher relaxation over the CrossDocked train pockets. Produces
# the distillation targets; these poses are never scored, so no eval step.
cd /gs/bs/tga-ohuelab/sakano/git/complex-tokenizer-bench
export T4TMPDIR="${T4TMPDIR:-$HOME/tmpdir}"; export TMPDIR="$T4TMPDIR"; mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
set -e
SB=/gs/bs/tga-ohuelab/sakano/git/sbdd-bench
PY=$SB/.venv/bin/python
IDFILE="${IDFILE:-data/trainpk_done.txt}"
ARM="${ARM:-separate_4096_trainpk}"
OUT="${OUT:-T_trainpk}"
echo "relax train pockets: $(wc -l < $IDFILE) targets"
xargs -P 38 -n 1 "$PY" scripts/relax_in_pocket.py \
    --arm "$ARM" --out-arm "$OUT" --index "$SB/data/train_pockets/index.json" \
    --contact-scale 1.10 --w-pkt 10 --w-tether 2.0 --w-uff 0.3 --w-internal 0 \
    --pocket-source receptor --targets < "$IDFILE"
echo "RELAX TRAINPK DONE ($OUT): $(find $SB/outputs/$OUT/own -name generated.sdf | wc -l) targets"

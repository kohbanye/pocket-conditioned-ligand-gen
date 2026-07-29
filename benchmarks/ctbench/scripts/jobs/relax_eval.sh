#!/bin/sh
#$ -cwd
#$ -l cpu_40=1
#$ -l h_rt=3:00:00
#$ -N ctb_relax_eval

# Pocket-aware clash relief on one arm's poses, then score them with vina_score
# (+ vina_min, to show how much local slack is left after the relief).
#
# vina_score is the only Vina metric that responds to our coordinates -- vina_dock
# re-docks from a bare XYZ and discards them -- so this is the pipeline that moves
# it: measured -2.922 -> -5.593 on the full 100-pocket set.
#
# Node: cpu_40 (40 CPU, 92 GB, no GPU). Runtime ~30 min relaxation + ~1 h scoring
#   for 100 targets x 100 molecules; h_rt 3 h is head-room.
#
#   qsub -g tga-ohuelab -p -3 -v ARM=sep4096_fin,OUT=sep4096_fin_rx \
#        scripts/jobs/relax_eval.sh

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/ctbench
export T4TMPDIR="${T4TMPDIR:-$HOME/tmpdir}"
export TMPDIR="$T4TMPDIR"
mkdir -p "$TMPDIR"
# One BLAS thread per worker: numpy otherwise spawns a thread per core in every
# worker and the process count explodes.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
set -e

ARM="${ARM:?set ARM}"
OUT="${OUT:?set OUT}"
CONTACT="${CONTACT:-1.05}"
WPKT="${WPKT:-10}"
WTETH="${WTETH:-0.5}"
WUFF="${WUFF:-0}"
# Intramolecular restraint on 1-2/1-3/1-4 distances. Without it the overlap
# relief spreads the atoms apart: PoseBusters validity 0.44 -> 0.11 and strain
# tripled, which is where the unrestrained score "gain" came from. At w=100 the
# geometry is preserved (validity 0.600 vs 0.592 baseline, strain unchanged) and
# vina_score still improves by 1.4 kcal/mol.
WINT="${WINT:-100}"
MODES="${MODES:-score min}"
WATT="${WATT:-0}"
MSTART="${MSTART:-1}"
STRANS="${STRANS:-1.0}"
SROT="${SROT:-15}"
IDFILE="${IDFILE:-data/target_ids.txt}"
# Restraint topology depth. 3 pins 1-2/1-3/1-4 distances, which freezes the
# torsions as well; 2 pins only bond lengths and bond angles and lets the
# dihedrals relax into the pocket -- the physically correct freedom for a
# rigid-geometry ligand.
INTPATH="${INTPATH:-3}"
PSRC="${PSRC:-pocket}"
SB=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/sbddbench
PY=$SB/.venv/bin/python

echo "relax $ARM -> $OUT contact=$CONTACT w_pkt=$WPKT w_tether=$WTETH w_uff=$WUFF w_internal=$WINT"
rm -rf "$SB/outputs/$OUT"
xargs -P 38 -n 1 "$PY" scripts/relax_in_pocket.py \
    --arm "$ARM" --out-arm "$OUT" \
    --contact-scale "$CONTACT" --w-pkt "$WPKT" --w-tether "$WTETH" --w-uff "$WUFF" --w-internal "$WINT" \
    --w-att "$WATT" --internal-path "$INTPATH" --pocket-source "$PSRC" --multi-start "$MSTART" --start-trans "$STRANS" --start-rot "$SROT" \
    --targets < "$IDFILE"
echo "relaxed $(find "$SB/outputs/$OUT/own" -name generated.sdf | wc -l) targets"

if [ -n "${SKIP_EVAL:-}" ]; then echo "RELAX ONLY DONE ($OUT)"; exit 0; fi

RES=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/ctbench/results/generation/$OUT
mkdir -p "$RES"
cd "$SB"
# shellcheck disable=SC2086
"$PY" scripts/run_evaluation.py --models own \
    --index "$SB/data/targets/index.json" \
    --out-dir "$SB/outputs/$OUT" --results "$RES" \
    --ids $(tr '\n' ' ' < /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/ctbench/"$IDFILE") \
    --dock-modes $MODES --dock-workers 38

echo "RELAX EVAL DONE ($OUT) -> $RES"

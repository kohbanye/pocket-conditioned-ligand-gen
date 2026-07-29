#!/bin/sh
#$ -cwd
#$ -l cpu_40=1
#$ -l h_rt=2:00:00
#$ -N ctb_score
# Score an already-refined arm with vina_score only, on 38 workers. Splitting the
# scoring out of the GPU job cuts the wall clock ~3x: the refiner needs a GPU for
# a few minutes, the docking needs many CPUs for much longer.
cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/sbddbench
export T4TMPDIR="${T4TMPDIR:-$HOME/tmpdir}"; export TMPDIR="$T4TMPDIR"; mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
set -e
OUT="${OUT:?set OUT}"
CT=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/ctbench
IDFILE="${IDFILE:-data/target_ids_ab.txt}"
RES=$CT/results/generation/$OUT
mkdir -p "$RES"
# shellcheck disable=SC2046
/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/.venv/bin/python scripts/run_evaluation.py --models own \
    --index data/targets/index.json --out-dir outputs/$OUT --results "$RES" \
    --ids $(tr '\n' ' ' < "$CT/$IDFILE") --dock-modes score --dock-workers 38
echo "SCORE DONE ($OUT)"

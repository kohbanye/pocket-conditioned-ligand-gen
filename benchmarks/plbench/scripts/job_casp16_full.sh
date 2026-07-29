#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=3:00:00
#$ -N casp16_full

# Full CASP16 reconstruction benchmark: every model, all 303 held-out complexes,
# with PoseBusters chemical-validity columns.
#
# Why this is a job and not an interactive run: the previous attempt shared a
# node with three running VQ-VAE trainings and every model died on CUDA OOM. A
# batch job gets its own GPU, which removes the contention entirely.
#
# Models:
#   own_allatom  every all-atom arm whose weights are trained past epoch 90 --
#                arms still training are skipped automatically, so re-submitting
#                this same script later picks up the new ones with no edits
#   esm3         protein structure tokenizer   (GPU, HF weights prefetched)
#   foldtoken    protein structure tokenizer   (GPU, own .venv-foldtoken)
#   token_mol    ligand torsion tokenizer
#
# PoseBusters dominates the runtime: ~22 s per drug-like ligand (one sampled
# molecule took 100 s), i.e. ~17 h serially across nine arms. It now runs in a
# process pool, and every arm checkpoints its own part file, so a walltime kill
# costs one arm rather than the whole run and --skip-done-arms resumes.
#
# node_q, not gpu_1, precisely because of that: the GPU is busy only for the
# brief VQ encode while PoseBusters is pure CPU, and node_q's 48 cores cut the
# checks from ~2.4 h (8 cores) to ~25 min. The higher billing coefficient is
# more than repaid by the shorter walltime.
#
# Ligand SDFs were rebuilt with OpenBabel first: RDKit's PDB reader had assigned
# every bond as single, which broke the chemistry features our own descriptor
# derives from the bond graph (hybridization wrong on 77% of atoms) and made the
# crystal reference itself fail PoseBusters. Re-running is what propagates that
# fix into the results.
#
# Submit with:
#   qsub -g tga-ohuelab -p -3 scripts/job_casp16_full.sh
#   qsub -g tga-ohuelab -p -3 -v LIMIT=4 scripts/job_casp16_full.sh   # smoke test

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/plbench || exit 1

# Deliberately NOT sourcing ~/.bashrc: under /bin/sh it terminates the script
# outright (verified -- a first attempt exited after 0.3 s with empty stdout and
# stderr, and exit status 0, which looks exactly like a successful no-op run).
# Nothing here needs it: every interpreter below is referenced by explicit path.

# HF_HOME explicitly: the ESM3 structure weights live on the group disk, not in
# $HOME/.cache, and the setting normally comes from ~/.zshrc -- which a job never
# reads. Combined with HF_HUB_OFFLINE this failed all 303 ESM3 samples with
# LocalEntryNotFoundError while the other models ran fine.
export HF_HOME=/gs/bs/tga-ohuelab/sakano/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

PY=.venv/bin/python
OUT=results/casp16.parquet
EXTRA=""
if [ -n "$LIMIT" ]; then
    # Smoke mode: a few complexes to a throwaway path, so a bad run can never
    # overwrite the real results file.
    EXTRA="--limit $LIMIT"
    OUT=results/casp16_smoke.parquet
fi

echo "[job] host=$(hostname) start=$(date -Is) out=$OUT"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

$PY scripts/run_reconstruction.py \
    --models own_allatom esm3 foldtoken token_mol \
    --dataset casp16 \
    --protein-scope pocket \
    --pb-valid \
    --skip-done-arms \
    --out "$OUT" \
    $EXTRA
status=$?

echo "[job] exit=$status end=$(date -Is)"
exit $status

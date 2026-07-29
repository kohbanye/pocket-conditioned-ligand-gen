#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=2:00:00
#$ -N casp16_prior

# Prior-work tokenizers on CASP16, everything except PoseBusters.
#
# Split out from job_casp16_full.sh because that job works through the nine
# all-atom ablation arms first and never reaches these models before its
# walltime. These four touch none of the all-atom outputs, so this can run
# alongside it without contending for a single file.
#
#   esm3        protein structure VQ-VAE   (GPU)
#   foldtoken   protein structure VQ-VAE   (GPU, own .venv-foldtoken)
#   token_mol   ligand torsion tokenizer
#
# No --pb-valid: PoseBusters averages ~22 s per ligand and dominates everything
# else, while ESM3 and FoldToken are protein-only and have no ligand row for it
# to score anyway. Every geometry metric (RMSD, TM-score, lDDT, lDDT-PLI,
# contact F1, clash, bond geometry) and the rate columns are cheap and included.
# PoseBusters for token_mol can be filled in later against the same
# sample set.
#
# Submit with:
#   qsub -g tga-ohuelab -p -3 scripts/job_casp16_priorwork.sh

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/plbench || exit 1

# NOT sourcing ~/.bashrc: under /bin/sh it terminates the script outright.
# HF_HOME explicitly: the ESM3 structure weights live on the group disk, not in
# $HOME/.cache, and the setting normally comes from ~/.zshrc -- which a job never
# reads. Combined with HF_HUB_OFFLINE this failed all 303 ESM3 samples with
# LocalEntryNotFoundError while the other models ran fine.
export HF_HOME=/gs/bs/tga-ohuelab/sakano/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

OUT=${OUT:-results/casp16_priorwork.parquet}
# '+'-separated, not space and not comma: qsub -v splits its value on
# whitespace (so "MODELS=a b" becomes a stray script argument) AND treats commas
# as the separator between variable assignments (so "MODELS=a,b" silently drops
# b and defines it as an empty variable -- which quietly ran only one model).
MODELS=$(echo "${MODELS_LIST:-own_allatom+esm3+foldtoken+token_mol}" | tr '+' ' ')

echo "[job] host=$(hostname) start=$(date -Is) out=$OUT"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

.venv/bin/python scripts/run_reconstruction.py \
    --models $MODELS \
    --dataset casp16 \
    --protein-scope pocket \
    --out "$OUT"
status=$?

echo "[job] exit=$status end=$(date -Is)"
exit $status

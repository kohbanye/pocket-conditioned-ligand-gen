#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=2:00:00
#$ -N casp16_finish

# The two pieces still missing from the CASP16 table, both with PoseBusters:
#
#   localframe_1tok  the last all-atom arm (the other eight are on disk as
#                    per-arm part files and --skip-done-arms leaves them alone)
#   confseq          ConfSeq (Xiong et al., Nat Mach Intell 2026) -- rule-based,
#                    no weights, the current state of the art for the
#                    "ligand in its own frame" family this paper argues against
#
# Runs as a job rather than on the interactive node because PoseBusters needs a
# process pool and the 2500-thread ceiling is per user and shared with every
# other job on that account -- filling it interactively killed this same work
# twice. node_q gives 48 dedicated cores; PLBENCH_PB_WORKERS lifts the
# conservative default that protects the shared case.
#
# Submit with:
#   qsub -g tga-ohuelab -p -3 scripts/job_casp16_finish.sh

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/plbench || exit 1

# NOT sourcing ~/.bashrc: under /bin/sh it terminates the script outright.
export HF_HOME=/gs/bs/tga-ohuelab/sakano/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PLBENCH_PB_WORKERS=40

echo "[job] host=$(hostname) start=$(date -Is) cores=$(nproc)"

.venv/bin/python scripts/run_reconstruction.py \
    --models own_allatom --allatom-arms localframe_1tok \
    --dataset casp16 --protein-scope full \
    --pb-valid --skip-done-arms \
    --out results/casp16.parquet
s1=$?
echo "[job] localframe_1tok exit=$s1"

.venv/bin/python scripts/run_reconstruction.py \
    --models confseq \
    --dataset casp16 --protein-scope full \
    --pb-valid \
    --out results/casp16_confseq.parquet
s2=$?
echo "[job] confseq exit=$s2"

echo "[job] end=$(date -Is)"
[ $s1 -eq 0 ] && [ $s2 -eq 0 ]

#!/bin/sh
#$ -cwd
#$ -l cpu_16=1
#$ -l h_rt=4:00:00
#$ -N casf_dl

# RTMScore + GenScore docking power on all 285 CASF-2016 targets (the DL SOTA
# baselines, sc8668). Uses the micromamba envs the setup agent built under
# baselines/. Each backend is ~4-threaded, serial over targets (~40-60 min);
# CASF-standard decoys-only (crystal native excluded), directly comparable to
# our head's honest 87.7%. Outputs baselines/casf_work/docking_power_{backend}.csv.

export PATH=$HOME/.local/bin:$PATH
cd /gs/bs/tga-ohuelab/sakano/git/baselines
set -e

OMP_NUM_THREADS=4 ./run_casf.sh genscore
OMP_NUM_THREADS=4 ./run_casf.sh rtmscore

echo "CASF DL BASELINES DONE"

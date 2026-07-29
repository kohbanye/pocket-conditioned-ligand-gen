#!/bin/sh
#$ -cwd
#$ -l cpu_40=1
#$ -l h_rt=3:00:00
#$ -N casf_vina

# AutoDock Vina docking-power baseline on all 285 CASF-2016 targets (~28k poses:
# obabel -> pdbqt -> vina --score_only, rank by affinity). CPU-only (no GPU),
# 40 workers. Same pose sets as our rescorer -> apples-to-apples docking power.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
set -e

.venv/bin/python scripts/eval_casf_vina.py \
    --workers 40 \
    --exclude-native \
    --out-csv outputs/casf/vina_decoyonly.csv

echo "CASF VINA DONE"

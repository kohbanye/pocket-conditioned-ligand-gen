#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=1:00:00
#$ -N diag_repr
cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
.venv/bin/python scripts/diag_casf_repr.py
echo "DIAG REPR DONE"

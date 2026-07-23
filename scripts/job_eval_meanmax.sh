#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=1:00:00
#$ -N eval_mm

# Recover the meanmax affinity result: job 8188969 trained the head fine but was
# killed by the 6h wall during eval. Re-run eval only against its saved ckpt.
cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
set -e
MLM=pocket-ligand-mlm/wxlhgqx3/checkpoints/mlm-e02-vl0.8199.ckpt
VQ="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
NORM=data/descriptor_cache_allatom/normalization_stats.pt
CKPT="pocket-ligand-rescore/3c58a53e/checkpoints/rescore-e06-vl0.6211.ckpt"
.venv/bin/python scripts/eval_casf_scoring.py \
    --mlm-ckpt "$MLM" --vqvae-ckpt "$VQ" --norm-stats "$NORM" \
    --rescore-ckpt "$CKPT" --pooling meanmax --affinity-head \
    --out-csv outputs/casf/affinity_kdki_meanmax.csv
echo "EVAL MEANMAX DONE"

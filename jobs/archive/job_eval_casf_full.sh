#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=4:00:00
#$ -N casf_eval

# Full CASF-2016 (285 targets) docking-power evaluation of our complex-token
# rescorer, for the comparison table vs Vina / RTMScore / GenScore. Runs the
# discriminative head and the zero-shot PLL, each ranking DECOYS-ONLY (the
# honest metric) and (for the head) also with the crystal native. Per-target
# CSVs under outputs/casf/.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

VQ="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
MLM="pocket-ligand-mlm/j90rlrgm/checkpoints/mlm-e01-vl0.8528.ckpt"
HEAD="pocket-ligand-rescore/7qamlrip/checkpoints/rescore-e01-vl0.1672.ckpt"
NS="data/descriptor_cache_allatom/normalization_stats.pt"
mkdir -p outputs/casf

echo "=== HEAD, decoys-only (honest) ==="
.venv/bin/python scripts/eval_casf_rescore.py --score-mode head --rescore-ckpt "$HEAD" \
    --mlm-ckpt "$MLM" --vqvae-ckpt "$VQ" --norm-stats "$NS" --exclude-native \
    --out-csv outputs/casf/head_decoyonly.csv

echo "=== HEAD, with crystal native ==="
.venv/bin/python scripts/eval_casf_rescore.py --score-mode head --rescore-ckpt "$HEAD" \
    --mlm-ckpt "$MLM" --vqvae-ckpt "$VQ" --norm-stats "$NS" \
    --out-csv outputs/casf/head_native.csv

echo "=== zero-shot PLL, decoys-only ==="
.venv/bin/python scripts/eval_casf_rescore.py --score-mode pll \
    --mlm-ckpt "$MLM" --vqvae-ckpt "$VQ" --norm-stats "$NS" --exclude-native \
    --out-csv outputs/casf/pll_decoyonly.csv

echo "CASF FULL EVAL DONE"

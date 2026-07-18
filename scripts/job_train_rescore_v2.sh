#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=6:00:00
#$ -N rescore_v2

# Iteration 1 of the win-vs-SOTA loop: bigger, more realistic decoy set
# (25k complexes x 20 rigid+CONFORMATIONAL decoys -> denser near-native boundary,
# closing the train/test distribution gap that caused our 12 near-miss + 7 gross
# failures) -> retrain the scoring head (warm-start from MLM epoch1). Then eval.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

VQ="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
NS="data/descriptor_cache_allatom/normalization_stats.pt"

.venv/bin/python scripts/tokenize_decoys.py --ckpt "$VQ" --norm-stats "$NS" \
    --casf-pdbs data/casf2016_pdbs.txt \
    --n-complexes 12000 --n-decoys 20 --out-dir data/lm_tokens_decoys_v2

.venv/bin/python scripts/train_rescore.py \
    --token-dir data/lm_tokens_decoys_v2 \
    --mlm-ckpt pocket-ligand-mlm/j90rlrgm/checkpoints/mlm-e01-vl0.8528.ckpt \
    --micro-batch-size 32 --num-workers 7 --max-epochs 15 --early-stop-patience 3 \
    --run-name rescore_head_v2

echo "RESCORE V2 DONE"

#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=6:00:00
#$ -N aff_eff

# Affinity head trained on ligand efficiency (pK / heavy-atom count) instead of
# raw pK, then evaluated by multiplying the prediction back by size.
#
# Why: the diagnosis pinned the head's weakness on riding molecular size (pred-MW
# corr 0.537 vs truth 0.496; MW-partialled discrimination 0.679 vs GenScore's
# 0.773). Efficiency strips the size trend from the target (pK-size corr 0.37),
# forcing the head onto contact quality. The all-atom tokenizer emits one token
# per heavy atom, so size = ligand token count -- no re-tokenization needed.
#
#   qsub -g tga-ohuelab scripts/job_aff_efficiency.sh <pooling> <token-dir> <tag>

POOL=${1:-mean}
TOKDIR=${2:-data/lm_tokens_affinity}
TAG=${3:-eff_mean}

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

MLM=pocket-ligand-mlm/wxlhgqx3/checkpoints/mlm-e02-vl0.8199.ckpt   # leak-free
VQ="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
NORM=data/descriptor_cache_allatom/normalization_stats.pt

RUN_NAME="rescore_aff_${TAG}"
CKPT_DIR="pocket-ligand-rescore/${RUN_NAME}/checkpoints"
rm -f "$CKPT_DIR"/rescore-e*.ckpt 2>/dev/null || true

echo "=== train efficiency head: pooling=$POOL tokens=$TOKDIR ==="
.venv/bin/python scripts/train_rescore.py \
    --token-dir "$TOKDIR" --mlm-ckpt "$MLM" \
    --run-name "$RUN_NAME" \
    --max-epochs 15 --micro-batch-size 32 --early-stop-patience 3 \
    --pooling "$POOL" --label-cap 13.0 --num-workers 8 --efficiency

CKPT=$(ls "$CKPT_DIR"/rescore-e*-vl*.ckpt 2>/dev/null \
       | sed -E 's/.*-vl([0-9.]+)\.ckpt/\1 &/' | sort -n | head -1 | cut -d' ' -f2-)
if [ -z "$CKPT" ]; then echo "ERROR: no checkpoint in $CKPT_DIR"; exit 1; fi
echo "=== best ckpt: $CKPT ==="

echo "=== CASF scoring/ranking (efficiency head, multiply back by size) ==="
.venv/bin/python scripts/eval_casf_scoring.py \
    --mlm-ckpt "$MLM" --vqvae-ckpt "$VQ" --norm-stats "$NORM" \
    --rescore-ckpt "$CKPT" --pooling "$POOL" --efficiency-head \
    --out-csv "outputs/casf/affinity_${TAG}.csv"

echo "AFF EFFICIENCY ${TAG} DONE"

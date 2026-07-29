#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=8:00:00
#$ -N aff_aux

# Affinity-aware encoder adaptation: fine-tune with a within-protein ranking
# loss PLUS a masked-LM regularizer. The head-only ranking loss collapsed the
# 99M encoder onto the 14k corpus (train/rank -> 0, no generalization); the MLM
# term keeps the pretrained structure representation intact so the ranking
# signal can adapt the encoder without memorizing. Tokenizer is unchanged.
#
#   qsub -g tga-ohuelab scripts/job_aff_mlmaux.sh <pooling> <tag> <rank-w> <mlm-w> [int-layers]

POOL=${1:-xattn}
TAG=${2:-aux_xattn}
RANKW=${3:-0.3}
MLMW=${4:-0.5}
NLAYERS=${5:-0}

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

MLM=pocket-ligand-mlm/wxlhgqx3/checkpoints/mlm-e02-vl0.8199.ckpt   # leak-free
VQ="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
NORM=data/descriptor_cache_allatom/normalization_stats.pt
TOKDIR=data/lm_tokens_affinity_grp   # has {split}.grp protein groups for ranking

RUN_NAME="rescore_aff_${TAG}"
CKPT_DIR="pocket-ligand-rescore/${RUN_NAME}/checkpoints"
rm -f "$CKPT_DIR"/rescore-e*.ckpt 2>/dev/null || true

echo "=== train affinity-aware head: pool=$POOL rank=$RANKW mlm=$MLMW int=$NLAYERS ==="
.venv/bin/python pipelines/train/head.py \
    --token-dir "$TOKDIR" --mlm-ckpt "$MLM" \
    --run-name "$RUN_NAME" \
    --max-epochs 20 --micro-batch-size 32 --early-stop-patience 4 \
    --pooling "$POOL" --label-cap 13.0 --num-workers 8 \
    --ranking-loss-weight "$RANKW" --complexes-per-batch 8 --max-per-group 4 \
    --mlm-aux-weight "$MLMW" --interaction-layers "$NLAYERS"

CKPT=$(ls "$CKPT_DIR"/rescore-e*-vl*.ckpt 2>/dev/null \
       | sed -E 's/.*-vl([0-9.]+)\.ckpt/\1 &/' | sort -n | head -1 | cut -d' ' -f2-)
if [ -z "$CKPT" ]; then echo "ERROR: no checkpoint in $CKPT_DIR"; exit 1; fi
echo "=== best ckpt: $CKPT ==="

echo "=== CASF scoring/ranking (affinity-aware head) ==="
.venv/bin/python scripts/eval_casf_scoring.py \
    --mlm-ckpt "$MLM" --vqvae-ckpt "$VQ" --norm-stats "$NORM" \
    --rescore-ckpt "$CKPT" --pooling "$POOL" --affinity-head \
    --interaction-layers "$NLAYERS" \
    --out-csv "outputs/casf/affinity_${TAG}.csv"

echo "AFF MLMAUX ${TAG} DONE"

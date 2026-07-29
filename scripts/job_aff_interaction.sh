#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=6:00:00
#$ -N aff_int

# Affinity head with N trainable transformer layers inserted over the tokens
# before pooling -- more capacity to re-model the pocket-ligand interface from
# the existing VQ tokens (the tokenizer is unchanged; only the head grows).
# Diagnosis: the MLM representation weakly encodes affinity (within-cluster
# distance-pK corr 0.19) and simple pooled heads plateau at ranking ~0.67; a
# deeper interaction head may extract more of the ranking signal.
#
#   qsub -g tga-ohuelab scripts/job_aff_interaction.sh <pooling> <token-dir> <tag> <n-layers>

POOL=${1:-mean}
TOKDIR=${2:-data/lm_tokens_affinity}
TAG=${3:-int_mean}
NLAYERS=${4:-2}

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

echo "=== train interaction head: pooling=$POOL layers=$NLAYERS tokens=$TOKDIR ==="
.venv/bin/python pipelines/train/head.py \
    --token-dir "$TOKDIR" --mlm-ckpt "$MLM" \
    --run-name "$RUN_NAME" \
    --max-epochs 15 --micro-batch-size 32 --early-stop-patience 3 \
    --pooling "$POOL" --label-cap 13.0 --num-workers 8 \
    --interaction-layers "$NLAYERS"

CKPT=$(ls "$CKPT_DIR"/rescore-e*-vl*.ckpt 2>/dev/null \
       | sed -E 's/.*-vl([0-9.]+)\.ckpt/\1 &/' | sort -n | head -1 | cut -d' ' -f2-)
if [ -z "$CKPT" ]; then echo "ERROR: no checkpoint in $CKPT_DIR"; exit 1; fi
echo "=== best ckpt: $CKPT ==="

echo "=== CASF scoring/ranking (interaction head) ==="
.venv/bin/python scripts/eval_casf_scoring.py \
    --mlm-ckpt "$MLM" --vqvae-ckpt "$VQ" --norm-stats "$NORM" \
    --rescore-ckpt "$CKPT" --pooling "$POOL" --affinity-head \
    --interaction-layers "$NLAYERS" \
    --out-csv "outputs/casf/affinity_${TAG}.csv"

echo "AFF INTERACTION ${TAG} DONE"

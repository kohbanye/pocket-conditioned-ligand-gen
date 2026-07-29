#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=4:00:00
#$ -N pose_te

# Self-contained pose-head experiment: train an RMSD head on a decoy corpus,
# pick its own best checkpoint, then run the CASF-2016 docking-power eval and
# dump per-pose scores. One job, because the checkpoint only exists after
# training (a separate eval job cannot be chained to it).
#
#   qsub -g tga-ohuelab scripts/job_pose_train_eval.sh <pooling> <token-dir> <tag> [int-layers] [mlm-ckpt] [extra-train-args]
#     pooling    : mean | meanmax | attn | xattn | pairsum
#     token-dir  : data/lm_tokens_decoys_v2
#     tag        : output name -> outputs/casf/{head,pose_scores}_<tag>.csv
#     int-layers : trainable transformer layers before pooling (0 = none)
#     mlm-ckpt   : encoder warm-start (default j90rlrgm e01, the best docking backbone)
#
# Why these knobs: the mean-pooled head tops out at 89.1% top1<2A while RTMScore
# (94.4) and GenScore (91.5) model protein-ligand contacts explicitly. xattn and
# pairsum give the head that missing pairwise-interaction inductive bias, and the
# interaction layers let it re-model the interface from the tokens -- all without
# touching the tokenizer or the pretrained LM.

POOL=${1:?"arg1: mean|meanmax|attn|xattn|pairsum"}
TOKDIR=${2:?"arg2: token dir"}
TAG=${3:?"arg3: output tag"}
NINT=${4:-0}
MLM=${5:-pocket-ligand-mlm/j90rlrgm/checkpoints/mlm-e01-vl0.8528.ckpt}
# Everything after arg 5 is passed through to train_rescore.py. SGE splits job
# arguments on whitespace, so a quoted "--flag value" would arrive as two args;
# shifting and using "$@" handles any number of them.
shift 5 2>/dev/null || shift $#

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

VQ="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
NORM=data/descriptor_cache_allatom/normalization_stats.pt

RUN_NAME="rescore_pose_${TAG}"
CKPT_DIR="pocket-ligand-rescore/${RUN_NAME}/checkpoints"
rm -f "$CKPT_DIR"/rescore-e*.ckpt 2>/dev/null || true

echo "=== train pose head: pooling=$POOL int=$NINT tokens=$TOKDIR mlm=$MLM ==="
# shellcheck disable=SC2086
.venv/bin/python pipelines/train/head.py \
    --token-dir "$TOKDIR" \
    --mlm-ckpt "$MLM" \
    --run-name "$RUN_NAME" \
    --max-epochs 15 --micro-batch-size 32 --early-stop-patience 3 \
    --pooling "$POOL" --interaction-layers "$NINT" --num-workers 8 "$@"

# best checkpoint = smallest val loss encoded in the filename
CKPT=$(ls "$CKPT_DIR"/rescore-e*-vl*.ckpt 2>/dev/null \
       | sed -E 's/.*-vl([0-9.]+)\.ckpt/\1 &/' | sort -n | head -1 | cut -d' ' -f2-)
if [ -z "$CKPT" ]; then echo "ERROR: no checkpoint in $CKPT_DIR"; exit 1; fi
echo "=== best ckpt: $CKPT ==="

echo "=== CASF docking power (decoys only) ==="
.venv/bin/python scripts/eval_casf_rescore.py --score-mode head \
    --pooling "$POOL" --interaction-layers "$NINT" \
    --rescore-ckpt "$CKPT" --mlm-ckpt "$MLM" \
    --vqvae-ckpt "$VQ" --norm-stats "$NORM" \
    --exclude-native \
    --out-csv "outputs/casf/head_${TAG}.csv" \
    --dump-scores "outputs/casf/pose_scores_${TAG}.csv"

echo "POSE TRAIN+EVAL ${TAG} DONE"

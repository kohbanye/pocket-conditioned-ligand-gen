#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=9:00:00
#$ -N pose_v8

# Train + eval a pose head on the v8 corpus (3.3x more complexes than v2, plus
# the conformer decoy class). Merges the tokenize shards first if that has not
# been done yet -- the merge is idempotent, and chaining this job on the
# tokenize array with -hold_jid makes the whole corpus->head->CASF path
# unattended.
#
#   qsub -g tga-ohuelab -hold_jid <tokenize-array-id> scripts/job_pose_v8.sh <tag> [extra train args...]
#     tag: output name -> outputs/casf/{head,pose_scores}_<tag>.csv

TAG=${1:?"arg1: output tag"}
shift

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

TOK=data/lm_tokens_decoys_v8
# Sibling jobs released by the same -hold_jid start together, so guard the merge
# with an atomic mkdir: one job merges, the others wait for its .done marker.
if [ ! -f "$TOK/train.bin" ]; then
    if mkdir "$TOK/.merge.lock" 2>/dev/null; then
        .venv/bin/python scripts/concat_decoy_shards.py "$TOK"
        touch "$TOK/.merge.done"
    else
        while [ ! -f "$TOK/.merge.done" ]; do sleep 20; done
    fi
fi

MLM=pocket-ligand-mlm/j90rlrgm/checkpoints/mlm-e01-vl0.8528.ckpt
VQ="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
RUN="rescore_pose_${TAG}"
CKPT_DIR="pocket-ligand-rescore/${RUN}/checkpoints"
rm -f "$CKPT_DIR"/rescore-e*.ckpt 2>/dev/null || true

echo "=== train pose head on v8: tag=$TAG extra=$* ==="
.venv/bin/python scripts/train_rescore.py \
    --token-dir "$TOK" --mlm-ckpt "$MLM" --run-name "$RUN" \
    --max-epochs 8 --micro-batch-size 32 --early-stop-patience 2 \
    --pooling mean --num-workers 8 "$@"

CKPT=$(ls "$CKPT_DIR"/rescore-e*-vl*.ckpt 2>/dev/null \
       | sed -E 's/.*-vl([0-9.]+)\.ckpt/\1 &/' | sort -n | head -1 | cut -d' ' -f2-)
if [ -z "$CKPT" ]; then echo "ERROR: no checkpoint in $CKPT_DIR"; exit 1; fi
echo "=== best ckpt: $CKPT ==="

.venv/bin/python scripts/eval_casf_rescore.py --score-mode head --pooling mean \
    --rescore-ckpt "$CKPT" --mlm-ckpt "$MLM" --vqvae-ckpt "$VQ" \
    --norm-stats data/descriptor_cache_allatom/normalization_stats.pt --exclude-native \
    --out-csv "outputs/casf/head_${TAG}.csv" \
    --dump-scores "outputs/casf/pose_scores_${TAG}.csv"

echo "POSE V8 ${TAG} DONE"

#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=20:00:00
#$ -N pose_v12

# Train + eval a pose head on the v12 corpus = v8 (21.9k complexes, 6-50 heavy
# atoms) CONCATENATED with the large-ligand corpus (35-200 heavy atoms).
#
# Why: broken down by ligand size, the v8 head is the best method on CASF for
# ligands under 30 heavy atoms (95.3% vs RTMScore 94.0 / GenScore 92.1) but wins
# only 1 of 6 targets above 50 heavy atoms (RTMScore 83.3, GenScore 66.7). The
# decoy corpus was capped at --max-heavy 50, so the head had never seen a ligand
# that large. Those 6 targets are worth ~1.8 points of Top-1@2A on their own.
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

TOK=data/lm_tokens_decoys_v12
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

echo "=== train pose head on v12: tag=$TAG extra=$* ==="
.venv/bin/python scripts/train_rescore.py \
    --token-dir "$TOK" --mlm-ckpt "$MLM" --run-name "$RUN" \
    --max-epochs 4 --micro-batch-size 24 --early-stop-patience 2 \
    --block-size 640 \
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

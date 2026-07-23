#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=12:00:00
#$ -N aff_te

# Self-contained affinity experiment: train a head, find its own best checkpoint,
# then run the CASF-2016 scoring/ranking-power eval. One job so it needs no
# session -- the checkpoint dir is generated at train time and cannot be chained
# to a separate eval job.
#
#   qsub -g tga-ohuelab scripts/job_aff_train_eval.sh <pooling> <token-dir> <tag> [rank-w] [max-per-group]
#     pooling      : mean | meanmax | attn
#     token-dir    : data/lm_tokens_affinity_grp  (has {split}.grp = protein ids)
#     tag          : output name, e.g. grp_attn_rank
#     rank-w       : within-protein ranking loss weight (0 = off, default)
#     max-per-group: ligands drawn per protein per batch (needs rank-w > 0)
#
# Diagnosis driving this: our head leans on molecular size (MW corr 0.537 vs
# truth 0.496; GenScore 0.411) and its MW-partialled discrimination is 0.679 vs
# GenScore's 0.773 -- i.e. it averages "what molecule is this" rather than
# focusing on the contacts. attn pooling lets it weight contact-relevant atoms;
# the ranking loss attacks the same confound from the label side, forcing the
# head to separate ligands of the SAME protein (where MW barely varies) instead
# of riding the cross-protein size trend. That is also what ranking power scores.

POOL=${1:?"arg1: mean|meanmax|attn"}
TOKDIR=${2:?"arg2: token dir"}
TAG=${3:?"arg3: output tag"}
RANKW=${4:-0}
MAXPG=${5:-4}
# arg6: optional MLM backbone override. Default is the leak-free wxlhgqx3; pass
# j90rlrgm (leak-benign, trained on 4.6x more interface data) to test whether the
# richer backbone lifts affinity the way it lifted docking (+5%). The MLM never
# sees affinity labels, so a pK leak is impossible; a structure-memorization
# advantage is checked separately with a leaked/clean CASF split.
MLM_OVERRIDE=${6:-}
# arg7: "freeze" to train only pooling+head. Pairs with a ranking loss: the
# end-to-end ranking run memorized the 14k-doc corpus (train/rank -> 0); freezing
# the 99M encoder drops trainable params to ~0.6M so the ordering can generalize.
FREEZE=${7:-}

RANK_ARGS=""
if [ "$RANKW" != "0" ]; then
    # complexes_per_batch = proteins/batch; max_per_group = ligands/protein.
    # 8 x 4 = 32 docs, matching the micro-batch the no-ranking runs used.
    RANK_ARGS="--ranking-loss-weight $RANKW --complexes-per-batch 8 --max-per-group $MAXPG"
fi
if [ "$FREEZE" = "freeze" ]; then
    RANK_ARGS="$RANK_ARGS --freeze-encoder"
fi

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

MLM=${MLM_OVERRIDE:-pocket-ligand-mlm/wxlhgqx3/checkpoints/mlm-e02-vl0.8199.ckpt}  # default: leak-free
VQ="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
NORM=data/descriptor_cache_allatom/normalization_stats.pt

# Deterministic, per-experiment checkpoint dir (train_rescore.py pins it to the
# run name). No run-dir discovery -> no race when this job runs alongside its
# siblings. Clear any stale checkpoints from an earlier attempt at this tag so
# the "best val loss" pick below can't select a leftover.
RUN_NAME="rescore_aff_${TAG}"
CKPT_DIR="pocket-ligand-rescore/${RUN_NAME}/checkpoints"
rm -f "$CKPT_DIR"/rescore-e*.ckpt 2>/dev/null || true

echo "=== train affinity head: pooling=$POOL tokens=$TOKDIR rank=$RANKW ==="
.venv/bin/python scripts/train_rescore.py \
    --token-dir "$TOKDIR" \
    --mlm-ckpt "$MLM" \
    --run-name "$RUN_NAME" \
    --max-epochs 15 --micro-batch-size 32 --early-stop-patience 3 \
    --pooling "$POOL" --label-cap 13.0 --num-workers 8 $RANK_ARGS

# best checkpoint = smallest val loss encoded in the filename
CKPT=$(ls "$CKPT_DIR"/rescore-e*-vl*.ckpt 2>/dev/null \
       | sed -E 's/.*-vl([0-9.]+)\.ckpt/\1 &/' | sort -n | head -1 | cut -d' ' -f2-)
if [ -z "$CKPT" ]; then echo "ERROR: no checkpoint in $CKPT_DIR"; exit 1; fi
echo "=== best ckpt: $CKPT ==="

echo "=== CASF scoring/ranking power ==="
.venv/bin/python scripts/eval_casf_scoring.py \
    --mlm-ckpt "$MLM" --vqvae-ckpt "$VQ" --norm-stats "$NORM" \
    --rescore-ckpt "$CKPT" --pooling "$POOL" --affinity-head \
    --out-csv "outputs/casf/affinity_${TAG}.csv"

echo "AFF TRAIN+EVAL ${TAG} DONE"

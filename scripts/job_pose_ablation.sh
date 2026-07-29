#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=12:00:00
#$ -N pose_abl

# Tokenizer ablation: train one ensVAL3 member on a given decoy corpus + MLM
# backbone, then run the CASF-2016 docking-power eval and dump per-pose scores.
#
#   qsub -g tga-ohuelab -p -3 scripts/job_pose_ablation.sh <token-dir> <mlm-ckpt> <tag> <joint|sep4096> [extra train args...]
#
# "sep4096" means the tokens come from a protein-only and a ligand-only VQ-VAE
# (4096 codes each) unified into one 8192-code space, so both the head and the
# eval need the separate VQ checkpoints; the MLM's code space is the combined
# one either way, hence --atom-codebook-size 8192 in both modes.

TOK=${1:?"arg1: token dir"}
MLM=${2:?"arg2: mlm ckpt"}
TAG=${3:?"arg3: output tag"}
VQMODE=${4:?"arg4: joint|sep4096"}
shift 4

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

RUN="rescore_abl_${TAG}"
CKPT_DIR="pocket-ligand-rescore/${RUN}/checkpoints"

if [ "$VQMODE" = "sep4096" ]; then
    VQ_ARGS="--separate-protein-ckpt pocket-ligand-vqvae/protein-vqvae-4096/checkpoints/last.ckpt \
             --separate-protein-norm data/descriptor_cache_allatom/normalization_stats_protein.pt \
             --separate-ligand-ckpt pocket-ligand-vqvae/ligand-vqvae-4096/checkpoints/last.ckpt \
             --separate-ligand-norm data/descriptor_cache_allatom/normalization_stats_ligand.pt \
             --codebook-size 4096"
else
    VQ_ARGS="--vqvae-ckpt pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt \
             --norm-stats data/descriptor_cache_allatom/normalization_stats.pt --codebook-size 8192"
fi

# Resume if an earlier attempt left a checkpoint (the node can lose the job).
RESUME=$(ls "$CKPT_DIR"/rescore-e*-vl*.ckpt 2>/dev/null | sort | tail -1)
RESUME_ARG=""
[ -n "$RESUME" ] && RESUME_ARG="--resume-from $RESUME" && echo "resuming from $RESUME"

echo "=== train: tok=$TOK mlm=$MLM mode=$VQMODE extra=$* ==="
# shellcheck disable=SC2086
.venv/bin/python pipelines/train/head.py \
    --token-dir "$TOK" --mlm-ckpt "$MLM" --run-name "$RUN" \
    --atom-codebook-size 8192 --block-size 640 \
    --micro-batch-size 24 --num-workers 8 --pooling mean \
    --max-epochs 4 --early-stop-patience 2 $RESUME_ARG "$@"

CKPT=$(ls "$CKPT_DIR"/rescore-e*-vl*.ckpt 2>/dev/null \
       | sed -E 's/.*-vl([0-9.]+)\.ckpt/\1 &/' | sort -n | head -1 | cut -d' ' -f2-)
if [ -z "$CKPT" ]; then echo "ERROR: no checkpoint in $CKPT_DIR"; exit 1; fi
echo "=== best ckpt: $CKPT ==="

# shellcheck disable=SC2086
.venv/bin/python scripts/eval_casf_rescore.py --score-mode head \
    --rescore-ckpt "$CKPT" --mlm-ckpt "$MLM" $VQ_ARGS --exclude-native \
    --out-csv "outputs/casf/head_${TAG}.csv" \
    --dump-scores "outputs/casf/pose_scores_${TAG}.csv"

echo "POSE ABLATION ${TAG} DONE"

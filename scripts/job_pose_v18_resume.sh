#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=12:00:00
#$ -N pose_v18r

# Resume the v18 pose head (best recipe: v18 corpus + per-atom displacement +
# listwise + label-cap 4) from whatever checkpoint the interrupted local run
# left behind, then run the CASF-2016 eval and dump per-pose scores.
#
# v18 is the LARGE-LIGAND head of the size-routed scorer: ligands with <= 50
# heavy atoms go to the small/medium ensemble, larger ones come here. Only its
# large-ligand band matters, which is why its weaker 31-50 band is irrelevant.
#
#   qsub -g tga-ohuelab -p -3 scripts/job_pose_v18_resume.sh

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

TOK=data/lm_tokens_decoys_v18
MLM=pocket-ligand-mlm/j90rlrgm/checkpoints/mlm-e01-vl0.8528.ckpt
VQ="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
RUN=rescore_pose_p_v18_best
CKPT_DIR="pocket-ligand-rescore/${RUN}/checkpoints"

# Highest-epoch checkpoint = furthest along; resume from it if one exists.
RESUME=$(ls "$CKPT_DIR"/rescore-e*-vl*.ckpt 2>/dev/null | sort | tail -1)
RESUME_ARG=""
if [ -n "$RESUME" ]; then
    echo "=== resuming from $RESUME ==="
    RESUME_ARG="--resume-from $RESUME"
else
    echo "=== no checkpoint found; training from scratch ==="
fi

# shellcheck disable=SC2086
.venv/bin/python pipelines/train/head.py \
    --token-dir "$TOK" --mlm-ckpt "$MLM" --run-name "$RUN" \
    --max-epochs 3 --micro-batch-size 24 --early-stop-patience 1 \
    --block-size 640 --pooling mean --num-workers 8 \
    --atom-aux-weight 1.0 --listwise-weight 1.0 --label-cap 4.0 \
    --complexes-per-batch 3 --max-per-group 8 $RESUME_ARG

# best checkpoint = smallest val loss encoded in the filename
CKPT=$(ls "$CKPT_DIR"/rescore-e*-vl*.ckpt 2>/dev/null \
       | sed -E 's/.*-vl([0-9.]+)\.ckpt/\1 &/' | sort -n | head -1 | cut -d' ' -f2-)
if [ -z "$CKPT" ]; then echo "ERROR: no checkpoint in $CKPT_DIR"; exit 1; fi
echo "=== best ckpt: $CKPT ==="

.venv/bin/python scripts/eval_casf_rescore.py --score-mode head \
    --rescore-ckpt "$CKPT" --mlm-ckpt "$MLM" --vqvae-ckpt "$VQ" \
    --norm-stats data/descriptor_cache_allatom/normalization_stats.pt --exclude-native \
    --out-csv outputs/casf/head_p_v18_best.csv \
    --dump-scores outputs/casf/pose_scores_p_v18_best.csv

echo "POSE V18 RESUME DONE"

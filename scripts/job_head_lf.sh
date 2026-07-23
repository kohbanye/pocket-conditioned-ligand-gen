#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=2:00:00
#$ -N head_lf

# Retrain one pose-scoring head on the LEAK-FREE MLM backbone (wxlhgqx3 e02,
# masked val 0.8199 -- better than the leaky j90rlrgm 0.853 and honestly held
# out from CASF). Pooling variant passed as $1 (mean|meanmax|attn); the diverse
# pooling heads + Vina form the consensus ensemble.
#   qsub -g tga-ohuelab scripts/job_head_lf.sh meanmax

POOL=${1:?"arg1 must be mean|meanmax|attn"}

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python scripts/train_rescore.py \
    --token-dir data/lm_tokens_decoys_v2 \
    --mlm-ckpt pocket-ligand-mlm/wxlhgqx3/checkpoints/mlm-e02-vl0.8199.ckpt \
    --run-name "rescore_${POOL}_lf" \
    --max-epochs 12 --micro-batch-size 32 --early-stop-patience 3 \
    --pooling "$POOL" --num-workers 8

echo "HEAD ${POOL} LEAK-FREE DONE"

#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=6:00:00
#$ -N pl_sep4096

# GEN ablation, SEPARATE-4096 arm, STANDALONE placement (stage 3/3). Mirror of
# job_placement_joint_only.sh. Run only if the AR-7961 ar_sep4096 chain does NOT
# complete placement itself (fullft likely fills the reservation). Inits from the
# separate-4096 fullft last.ckpt. Combined 4096+4096 -> cb 8192.
# ckpt -> pocket-ligand-lm/lm_placement_sep4096/. Submit with -p -3.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python scripts/train_lm.py \
    --token-dir data/lm_tokens_goodmix_sep4096 \
    --atom-codebook-size 8192 \
    --mask-prompt \
    --init-from pocket-ligand-lm/lm_fullft_sep4096/checkpoints/last.ckpt \
    --micro-batch-size 64 \
    --lr 5e-5 \
    --max-epochs 4 \
    --early-stop-patience 2 \
    --run-name lm_placement_sep4096

echo "SEP4096 PLACEMENT (standalone) DONE"

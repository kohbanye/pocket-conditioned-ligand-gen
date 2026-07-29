#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=6:00:00
#$ -N pl_joint2

# GEN ablation, JOINT arm, STANDALONE placement (stage 3/3). The 3-stage chain
# (pretrain->fullft->placement) exceeds a 24h reservation, and the AR-7941 joint
# chain was killed mid-fullft before placement ran. Its fullft last.ckpt (2
# epochs) exists, so run placement from it here. Matches separate_4096's placement
# stage (both from a ~2-epoch fullft, fair). Combined JOINT VQ -> cb 8192.
# ckpt -> pocket-ligand-lm/lm_placement_joint2/. Submit with -p -3 (max priority).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python pipelines/train/clm.py \
    --token-dir data/lm_tokens_goodmix_joint \
    --atom-codebook-size 8192 \
    --mask-prompt \
    --init-from pocket-ligand-lm/lm_fullft_joint2/checkpoints/last.ckpt \
    --micro-batch-size 64 \
    --lr 5e-5 \
    --max-epochs 4 \
    --early-stop-patience 2 \
    --run-name lm_placement_joint2

echo "JOINT PLACEMENT (standalone) DONE"

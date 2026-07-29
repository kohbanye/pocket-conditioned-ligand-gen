#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=23:00:00
#$ -N ar_joint

# RESERVATION chain (AR 7941, node_f x2 x ~24h). JOINT-tokenizer generation LM,
# all three stages run BACK-TO-BACK in one continuous node_f job (mirror of
# job_ar_chain_sep.sh). Retrained on the SAME enlarged 4.04B all-poses cache as
# the separate arm (matched token count 4,041,478,321) so the two ablation arms
# differ ONLY in the tokenizer. Single-book JOINT VQ -> --atom-codebook-size 8192.
# Stages:
#   1. pretrain  (mixed corpus, ~789M tok x3ep, ~2-3h)      -> lm_pretrain_joint2
#   2. fullft    (allatom_full_joint 4.04B tok x3ep,~16-18h)-> lm_fullft_joint2
#   3. placement (goodmix_joint ~286M tok x4ep, ~1-2h)      -> lm_placement_joint2
# Combined ~20-23h < 23:50. save_last per stage -> graceful walltime degradation.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

echo "=== AR JOINT stage 1/3: pretrain ==="
.venv/bin/python pipelines/train/clm.py \
    --token-dir data/lm_tokens_pretrain_mixed_joint \
    --atom-codebook-size 8192 \
    --micro-batch-size 64 \
    --max-epochs 3 \
    --run-name lm_pretrain_joint2

echo "=== AR JOINT stage 2/3: fullft ==="
.venv/bin/python pipelines/train/clm.py \
    --token-dir data/lm_tokens_allatom_full_joint \
    --atom-codebook-size 8192 \
    --mask-prompt \
    --init-from pocket-ligand-lm/lm_pretrain_joint2/checkpoints/last.ckpt \
    --micro-batch-size 64 \
    --lr 3e-4 \
    --max-epochs 3 \
    --early-stop-patience 2 \
    --run-name lm_fullft_joint2

echo "=== AR JOINT stage 3/3: placement ==="
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

echo "AR JOINT CHAIN DONE (pretrain -> fullft -> placement)"

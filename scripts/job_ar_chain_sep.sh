#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=23:00:00
#$ -N ar_sep

# RESERVATION chain (AR 7941, node_f x2 x ~24h). SEPARATE-tokenizer generation
# LM, all three stages run BACK-TO-BACK in one continuous node_f job so the
# reserved node is used without inter-stage re-scheduling gaps and a single h_rt
# (23:50, fits the ~24h AR window) covers the whole chain. Each stage's
# train_lm.py pins dirpath + save_last, so a walltime kill degrades gracefully to
# the latest last.ckpt (the next eval / stage reads it). Stages:
#   1. pretrain  (mixed corpus, ~789M tok x3ep, ~2-3h)     -> lm_pretrain_sep
#   2. fullft    (allatom_full_sep 4.04B tok x3ep, ~16-18h)-> lm_fullft_sep
#   3. placement (goodmix_sep ~286M tok x4ep, ~1-2h)       -> lm_placement_sep
# Combined ~20-23h < 23:50. set -e: if an earlier stage dies the job stops rather
# than warm-starting the next from a missing/garbage ckpt. All _sep caches exist;
# combined 16384-code space (--atom-codebook-size 16384).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

echo "=== AR SEP stage 1/3: pretrain ==="
.venv/bin/python scripts/train_lm.py \
    --token-dir data/lm_tokens_pretrain_mixed_sep \
    --atom-codebook-size 16384 \
    --micro-batch-size 64 \
    --max-epochs 3 \
    --run-name lm_pretrain_sep

echo "=== AR SEP stage 2/3: fullft ==="
.venv/bin/python scripts/train_lm.py \
    --token-dir data/lm_tokens_allatom_full_sep \
    --atom-codebook-size 16384 \
    --mask-prompt \
    --init-from pocket-ligand-lm/lm_pretrain_sep/checkpoints/last.ckpt \
    --micro-batch-size 64 \
    --lr 3e-4 \
    --max-epochs 3 \
    --early-stop-patience 2 \
    --run-name lm_fullft_sep

echo "=== AR SEP stage 3/3: placement ==="
.venv/bin/python scripts/train_lm.py \
    --token-dir data/lm_tokens_goodmix_sep \
    --atom-codebook-size 16384 \
    --mask-prompt \
    --init-from pocket-ligand-lm/lm_fullft_sep/checkpoints/last.ckpt \
    --micro-batch-size 64 \
    --lr 5e-5 \
    --max-epochs 4 \
    --early-stop-patience 2 \
    --run-name lm_placement_sep

echo "AR SEP CHAIN DONE (pretrain -> fullft -> placement)"

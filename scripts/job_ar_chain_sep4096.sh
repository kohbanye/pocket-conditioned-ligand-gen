#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=23:00:00
#$ -N ar_sep4096

# FAIR-ABLATION REDO (separate 4096+4096 -> combined 8192) of job_ar_chain_sep.sh.
# RESERVATION chain (node_f x ~24h). SEPARATE-4096-tokenizer generation LM, all
# three stages run BACK-TO-BACK in one continuous node_f job so the reserved node
# is used without inter-stage re-scheduling gaps and a single h_rt covers the whole
# chain. Each stage's train_lm.py pins dirpath + save_last, so a walltime kill
# degrades gracefully to the latest last.ckpt (the next eval / stage reads it).
# Stages:
#   1. pretrain  (mixed corpus, ~3ep)     -> lm_pretrain_sep4096
#   2. fullft    (allatom_full_sep4096, 3ep) -> lm_fullft_sep4096
#   3. placement (goodmix_sep4096, 4ep)   -> lm_placement_sep4096
# set -e: if an earlier stage dies the job stops rather than warm-starting the
# next from a missing/garbage ckpt. All _sep4096 caches must exist; combined
# 8192-code space (--atom-codebook-size 8192).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

echo "=== AR SEP4096 stage 1/3: pretrain ==="
.venv/bin/python scripts/train_lm.py \
    --token-dir data/lm_tokens_pretrain_mixed_sep4096 \
    --atom-codebook-size 8192 \
    --micro-batch-size 64 \
    --max-epochs 3 \
    --run-name lm_pretrain_sep4096

echo "=== AR SEP4096 stage 2/3: fullft ==="
.venv/bin/python scripts/train_lm.py \
    --token-dir data/lm_tokens_allatom_full_sep4096 \
    --atom-codebook-size 8192 \
    --mask-prompt \
    --init-from pocket-ligand-lm/lm_pretrain_sep4096/checkpoints/last.ckpt \
    --micro-batch-size 64 \
    --lr 3e-4 \
    --max-epochs 3 \
    --early-stop-patience 2 \
    --run-name lm_fullft_sep4096

echo "=== AR SEP4096 stage 3/3: placement ==="
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

echo "AR SEP4096 CHAIN DONE (pretrain -> fullft -> placement)"

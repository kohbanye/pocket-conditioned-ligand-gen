#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -N smlm_nocasf

# ABLATION (separate tokenizers): train the leak-free MLM backbone on the
# SEPARATE-tokenized nocasf corpus (data/lm_tokens_pretrain_nocasf_sep, built by
# job_build_mixed_pretrain_sep.sh). Combined code space => --atom-codebook-size
# 16384 (2x8192). Same arch/epochs as the joint MLM (wxlhgqx3) for an
# apples-to-apples tokenizer ablation. gpu_1 = 1 H100, ~8h/epoch x3.
# Chain: qsub -hold_jid sbuild_mix. ckpt -> pocket-ligand-mlm/mlm_nocasf_sep/.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

.venv/bin/python scripts/train_mlm.py \
    --token-dir data/lm_tokens_pretrain_nocasf_sep \
    --atom-codebook-size 16384 \
    --micro-batch-size 256 --num-workers 7 \
    --max-epochs 3 --early-stop-patience 2 \
    --run-name mlm_nocasf_sep

echo "SEPARATE MLM DONE"

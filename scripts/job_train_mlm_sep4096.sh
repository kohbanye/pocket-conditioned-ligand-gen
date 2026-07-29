#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -N smlm_nocasf4096

# FAIR-ABLATION REDO (separate 4096+4096 -> combined 8192): train the leak-free
# MLM backbone on the SEPARATE-4096-tokenized nocasf corpus
# (data/lm_tokens_pretrain_nocasf_sep4096, built by
# job_build_mixed_pretrain_sep4096.sh). Combined code space => --atom-codebook-size
# 8192 (2x4096), matching the joint MLM vocab. Same arch/epochs as the joint MLM
# for an apples-to-apples tokenizer ablation. gpu_1 = 1 H100, ~8h/epoch x3.
# Chain: qsub -hold_jid sbuild_mix4096. ckpt -> pocket-ligand-mlm/mlm_nocasf_sep4096/.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

.venv/bin/python scripts/train_mlm.py \
    --token-dir data/lm_tokens_pretrain_nocasf_sep4096 \
    --atom-codebook-size 8192 \
    --micro-batch-size 256 --num-workers 7 \
    --max-epochs 3 --early-stop-patience 2 \
    --run-name mlm_nocasf_sep4096

echo "SEPARATE4096 MLM DONE"

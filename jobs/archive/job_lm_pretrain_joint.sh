#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=8:00:00
#$ -N lm_pre_joint

# JOINT-tokenizer LM, stage 1/3 (mirror of job_lm_pretrain_sep.sh). Mixed-corpus
# pretraining of the all-atom pocket-conditioned LM on PLINDER protein-only +
# GEOM ligand-only (loss on all tokens -> learns p(pocket) and p(ligand)
# marginals), tokenized with the JOINT single-book VQ (8192 codes,
# --atom-codebook-size 8192, vocab 8199) instead of the separate combined
# 16384-code space. Same protocol/resource as the _sep stage 1 so the only
# difference is the tokenizer (the ablation). DDP over the 4 H100s. ~5 h.
# WANDB offline. Best by val/loss; save_last -> last.ckpt for stage 2 warm-start.
#
# hold_jid chain (wired at submit): holds on job_build_pretrain_mixed_joint.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline

.venv/bin/python pipelines/train/clm.py \
    --token-dir data/lm_tokens_pretrain_mixed_joint \
    --atom-codebook-size 8192 \
    --micro-batch-size 64 \
    --max-epochs 3 \
    --run-name lm_pretrain_joint2

echo "LM PRETRAIN JOINT DONE"

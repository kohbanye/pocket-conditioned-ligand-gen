#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=8:00:00
#$ -N lm_pre_sep

# SEPARATE-tokenizer LM, stage 1/3 (mirror of job_pretrain_lm_atom.sh, joint run
# vwvg82y2). Mixed-corpus pretraining of the all-atom pocket-conditioned LM on
# PLINDER protein-only + GEOM ligand-only (loss on all tokens -> learns p(pocket)
# and p(ligand) marginals), but tokenized with the SEPARATE protein-VQ +
# ligand-VQ combined 16384-code space (--atom-codebook-size 16384, vocab 16391)
# instead of the joint single-book VQ (8192). DDP over the 4 H100s. ~5 h.
# WANDB offline. Best by val/loss; save_last -> last.ckpt for stage 2 warm-start.
#
# hold_jid chain (wired at submit): holds on job_build_pretrain_mixed_sep.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline

.venv/bin/python scripts/train_lm.py \
    --token-dir data/lm_tokens_pretrain_mixed_sep \
    --atom-codebook-size 16384 \
    --micro-batch-size 64 \
    --max-epochs 3 \
    --run-name lm_pretrain_sep

echo "LM PRETRAIN SEP DONE"

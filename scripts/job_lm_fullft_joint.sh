#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=24:00:00
#$ -N lm_ff_joint

# JOINT-tokenizer LM, stage 2/3 (mirror of job_lm_fullft_sep.sh). Condition-only
# fine-tune on the FULL CrossDocked corpus (allatom_full_joint: all poses, x1
# rot), tokenized with the JOINT single-book VQ (8192 codes,
# --atom-codebook-size 8192, vocab 8199). Warm-started (weights only, fresh
# optimizer) from the stage-1 pretrain last.ckpt. --mask-prompt: loss only on the
# generated <l> ligand block. Same protocol/resource/lr/epochs as the _sep stage
# 2 so the only difference is the tokenizer. DDP over the 4 H100s. ~16-18 h (< 24 h).
# Best by held-out-pocket val/loss; save_last -> last.ckpt for stage 3.
#
# hold_jid chain (wired at submit): holds on job_lm_pretrain_joint AND on the
# allatom_full_joint tokenize/concat job (jconcat_atomfull /
# job_concat_atom_full_joint.sh).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline

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

echo "LM FULLFT JOINT DONE"

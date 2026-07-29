#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=24:00:00
#$ -N lm_ff_sep

# SEPARATE-tokenizer LM, stage 2/3 (mirror of job_finetune_lm_atom_full.sh, joint
# run awdya0s8). Condition-only fine-tune on the FULL CrossDocked corpus
# (allatom_full_sep, the _sep counterpart of allatom_full: all poses, x1 rot),
# tokenized with the SEPARATE VQ combined 16384-code space. Warm-started (weights
# only, fresh optimizer) from the stage-1 pretrain last.ckpt. --mask-prompt: loss
# only on the generated <l> ligand block. DDP over the 4 H100s. ~16-18 h (< 24 h).
# Best by held-out-pocket val/loss; save_last -> last.ckpt for stage 3.
#
# hold_jid chain (wired at submit): holds on job_lm_pretrain_sep AND on the
# allatom_full_sep tokenize job (stok_atomfull / job_tok_atom_full_sep.sh).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline

.venv/bin/python pipelines/train/clm.py \
    --token-dir data/lm_tokens_allatom_full_sep \
    --atom-codebook-size 16384 \
    --mask-prompt \
    --init-from pocket-ligand-lm/lm_pretrain_sep/checkpoints/last.ckpt \
    --micro-batch-size 64 \
    --lr 3e-4 \
    --max-epochs 3 \
    --early-stop-patience 2 \
    --run-name lm_fullft_sep

echo "LM FULLFT SEP DONE"

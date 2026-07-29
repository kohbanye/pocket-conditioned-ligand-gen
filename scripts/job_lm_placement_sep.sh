#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=8:00:00
#$ -N lm_pl_sep

# SEPARATE-tokenizer LM, stage 3/3 (mirror of job_finetune_lm_atom_placement.sh,
# joint run p6lpk7br). PLACEMENT re-finetune of the all-poses LM on
# placement-correct, CASF-held-out data only -- PLINDER complexes (diverse
# pockets) + CrossDocked good poses -- tokenized with the SEPARATE VQ combined
# 16384-code space (goodmix_sep). Warm-started (weights only, fresh optimizer)
# from the stage-2 fullft last.ckpt. Low LR (5e-5) adjusts placement without
# destroying the learned molecular geometry. --mask-prompt: loss only on the <l>
# block. Early stop on held-out val. DDP over the 4 H100s.
#
# hold_jid chain (wired at submit): holds on job_lm_fullft_sep AND on
# job_build_goodmix_sep.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline

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

echo "LM PLACEMENT SEP DONE"

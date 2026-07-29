#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=8:00:00
#$ -N lm_pl_joint

# JOINT-tokenizer LM, stage 3/3 (mirror of job_lm_placement_sep.sh). PLACEMENT
# re-finetune of the all-poses LM on placement-correct, CASF-held-out data only
# -- PLINDER complexes (diverse pockets) + CrossDocked good poses -- tokenized
# with the JOINT single-book VQ (8192 codes, --atom-codebook-size 8192, vocab
# 8199) (goodmix_joint). Warm-started (weights only, fresh optimizer) from the
# stage-2 fullft last.ckpt. Low LR (5e-5) adjusts placement without destroying
# the learned molecular geometry. --mask-prompt: loss only on the <l> block.
# Same protocol/resource/lr/epochs as the _sep stage 3 so the only difference is
# the tokenizer. Early stop on held-out val. DDP over the 4 H100s.
#
# hold_jid chain (wired at submit): holds on job_lm_fullft_joint AND on
# job_build_goodmix_joint.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline

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

echo "LM PLACEMENT JOINT DONE"

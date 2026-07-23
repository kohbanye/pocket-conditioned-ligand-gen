#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=8:00:00
#$ -N lm_place

# /loop iter1: PLACEMENT re-finetune of the all-poses all-atom LM.
#
# Rationale (measured): decoy poses carry VALID internal geometry but WRONG
# placement. Training on all 11.1M poses (awdya0s8) therefore lifted the
# intramolecular metrics (PB 0.03->0.17, validity 0.91->0.97, SA 6.31->5.47)
# but wrecked placement (raw Vina +3.93->+8.56). So: KEEP that model's internal
# geometry knowledge and re-finetune it on placement-correct data only --
#   PLINDER (427k docs, diverse pockets, CASF held out)  [fixes the 1,638-pocket
#   conditioning bottleneck]  +  CrossDocked good poses (min-only, 815k docs).
# Low LR (5e-5) so this adjusts placement without destroying the learned
# molecular geometry. Early stop on held-out val.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline

.venv/bin/python scripts/train_lm.py \
    --token-dir data/lm_tokens_atom_goodmix \
    --atom-codebook-size 8192 \
    --mask-prompt \
    --init-from pocket-ligand-lm/awdya0s8/checkpoints/lm-e01-vl4.9143.ckpt \
    --micro-batch-size 64 \
    --lr 5e-5 \
    --max-epochs 4 \
    --early-stop-patience 2 \
    --run-name lm_atom_placement_v1

echo "LM PLACEMENT FINETUNE DONE"

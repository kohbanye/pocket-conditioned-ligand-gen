#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=24:00:00
#$ -N lm_ft_full

# Stage 3 (all-atom, FULL): condition-only fine-tune of the all-atom
# pocket-conditioned LM on the FULL CrossDocked corpus (data/lm_tokens_allatom_full,
# 11.1M docs / 2.72B tokens -- 2x the legacy 2-codebook token budget, vs the
# starved 815k-doc set the undertrained 8a7umbru saw). Warm-started from the
# mixed pretraining checkpoint (vwvg82y2, val 2.18). --mask-prompt: loss only on
# the generated <l> ligand block. DDP over the 4 H100s (devices=auto). max_len
# 505 < block_size 512 -> no truncation. Best by held-out-pocket val/loss.
# ~5-6 h/epoch x 3 ~= 16-18 h (< 24 h). Goal: val loss well below 8a7umbru's ~4.3
# so generated molecules are internally coherent (PoseBusters recovers).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline

.venv/bin/python scripts/train_lm.py \
    --token-dir data/lm_tokens_allatom_full \
    --atom-codebook-size 8192 \
    --mask-prompt \
    --init-from pocket-ligand-lm/vwvg82y2/checkpoints/lm-e02-vl2.1796.ckpt \
    --micro-batch-size 64 \
    --lr 3e-4 \
    --max-epochs 3 \
    --early-stop-patience 2 \
    --run-name lm_finetune_atom_full_v1

echo "LM FINETUNE ATOM FULL DONE"

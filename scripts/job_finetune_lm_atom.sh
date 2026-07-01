#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=4:00:00
#$ -N lm_finetune

# Condition-only fine-tuning of the all-atom pocket-conditioned LM on CrossDocked
# complexes (data/lm_tokens_allatom, 211M tokens), warm-started from the mixed
# pretraining checkpoint. --mask-prompt trains only the generated <l> ligand
# block (loss masked on the <p> pocket prompt). DDP over 4 H100s; ~0.5 h/epoch
# x 3 = ~1.5 h. WANDB offline; best by val/loss. Has a test split -> final test.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline

.venv/bin/python scripts/train_lm.py \
    --token-dir data/lm_tokens_allatom \
    --atom-codebook-size 8192 \
    --mask-prompt \
    --init-from pocket-ligand-lm/vwvg82y2/checkpoints/lm-e02-vl2.1796.ckpt \
    --micro-batch-size 64 \
    --max-epochs 3 \
    --lr 3e-4 \
    --run-name lm_finetune_atom_v1

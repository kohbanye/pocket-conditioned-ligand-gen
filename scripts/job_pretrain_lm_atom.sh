#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=8:00:00
#$ -N lm_pretrain

# Mixed-corpus pretraining of the all-atom pocket-conditioned LM (~0.3B Qwen3,
# atom vocab 8199) on PLINDER protein-only + GEOM ligand-only (812M tokens,
# loss on all tokens -> learns p(pocket) and p(ligand) marginals). DDP over the
# 4 H100s. ~1.6 h/epoch x 3 = ~5 h. WANDB offline (headless). Best by val/loss.
# Fine-tune (CrossDocked complexes, --mask-prompt) is a separate job.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline

.venv/bin/python scripts/train_lm.py \
    --token-dir data/lm_tokens_pretrain_mixed \
    --atom-codebook-size 8192 \
    --micro-batch-size 64 \
    --max-epochs 3 \
    --run-name lm_pretrain_atom_v1

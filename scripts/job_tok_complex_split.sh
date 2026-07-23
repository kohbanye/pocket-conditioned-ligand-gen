#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=4:00:00
#$ -N cplx_split

# B4a of the split-codebook LM pipeline (runs in PARALLEL with the B3 pretrain;
# independent of it): build the condition-only FINE-TUNE corpus with the split
# atom VQ. Two sources, both -> 2-range LMVocab:
#   1) CrossDocked complexes, pocket-split (held-out-pocket val, cap 32/pocket,
#      x4 rot) -- the same recipe as the single-book finetune corpus.
#   2) PLINDER drug-like complexes (x2 rot, MW 150-600) for pocket diversity.
# Then concatenate into data/lm_tokens_finetune_split (train + held-out val).
# node_f: GPU encode + 40 CPU workers for PLINDER zip streaming. WANDB offline.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

CKPT="pocket-ligand-vqvae/ix6q6po0/checkpoints/atomvqvae-epoch=43-val/atom_coord=0.0632.ckpt"
NORM="data/descriptor_cache_allatom/normalization_stats.pt"

# 1) CrossDocked complexes (pocket-split held-out val).
.venv/bin/python scripts/tokenize_dataset_atom.py \
    --split-codebook --ckpt "$CKPT" --norm-stats "$NORM" \
    --source-types cdonly --pocket-split --max-per-pocket 32 --num-rotations 4 \
    --out-dir data/lm_tokens_complex_cd_split

# 2) PLINDER drug-like complexes.
.venv/bin/python scripts/tokenize_plinder_protein.py \
    --complex --split-codebook --ckpt "$CKPT" --norm-stats "$NORM" \
    --num-rotations 2 --num-workers 40 --batch-size 256 --mw-min 150 --mw-max 600 \
    --out-dir data/lm_tokens_complex_plinder_split

# 3) Combine (PLINDER + CrossDocked; train + held-out val).
.venv/bin/python scripts/build_mixed_pretrain_cache.py \
    --inputs data/lm_tokens_complex_plinder_split data/lm_tokens_complex_cd_split \
    --out-dir data/lm_tokens_finetune_split

echo "SPLIT FINETUNE CORPUS DONE"

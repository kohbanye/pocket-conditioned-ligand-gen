#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=8:00:00
#$ -N lm_pretrain_split

# B3 of the split-codebook LM pipeline: combine the GEOM (ligand) + PLINDER
# (protein) split token caches into a mixed pretrain corpus, then pretrain the
# 2-range LM (all-token loss) -- mirrors the single-book pretrain (vwvg82y2)
# but with the split VQ, so the finetune comparison isolates the tokenizer.
# NO --atom-codebook-size => LigandLMConfig default 2-range vocab (7 + 8192 +
# 4096 = 12295), matching the --split-codebook token caches. 4-GPU DDP. WANDB
# offline.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python scripts/build_mixed_pretrain_cache.py \
    --inputs data/lm_tokens_geom_split data/lm_tokens_protein_plinder_split \
    --out-dir data/lm_tokens_pretrain_split

.venv/bin/python scripts/train_lm.py \
    --token-dir data/lm_tokens_pretrain_split \
    --micro-batch-size 64 \
    --max-epochs 3 \
    --run-name lm_pretrain_split_v1

echo "SPLIT PRETRAIN DONE"

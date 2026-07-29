#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=8:00:00
#$ -N atom_lm_ft

# Self-contained conditional-finetune pipeline (survives client disconnect):
#   1) tokenize PLINDER complexes (drug-like, <p>pocket</p><l>ligand</l>)
#   2) combine with the pocket-split CrossDocked complexes -> finetune cache
#   3) condition-only finetune from the mixed-pretrain ckpt, held-out-pocket val
#      + EarlyStopping (picks a generalising model, not the most-overfit epoch).
# node_f: step 1 uses 1 GPU + many CPU workers; step 3 is 4-GPU DDP.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

CKPT="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"

# 1) PLINDER complex tokenization (~215k drug-like pocket-ligand pairs).
.venv/bin/python pipelines/corpora/tokenize_plinder.py --complex --ckpt "$CKPT" \
    --norm-stats data/descriptor_cache_allatom/normalization_stats.pt \
    --num-rotations 2 --num-workers 40 --batch-size 256 \
    --mw-min 150 --mw-max 600 \
    --out-dir data/lm_tokens_complex_plinder

# 2) Combine PLINDER + CrossDocked complexes (train + val, held-out pockets).
.venv/bin/python pipelines/corpora/mix.py \
    --inputs data/lm_tokens_complex_plinder data/lm_tokens_complex_crossdocked \
    --out-dir data/lm_tokens_finetune_mixed

# 3) Condition-only fine-tune (held-out-pocket val -> generalising model).
.venv/bin/python pipelines/train/clm.py \
    --token-dir data/lm_tokens_finetune_mixed \
    --atom-codebook-size 8192 \
    --mask-prompt \
    --init-from pocket-ligand-lm/vwvg82y2/checkpoints/lm-e02-vl2.1796.ckpt \
    --micro-batch-size 64 \
    --max-epochs 10 \
    --lr 3e-4 \
    --early-stop-patience 2 \
    --run-name lm_finetune_mixed_v1

echo "PIPELINE DONE"

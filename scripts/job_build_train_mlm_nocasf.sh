#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -N mlm_nocasf

# Leak-free MLM backbone: assemble the CASF-excluded corpus (clean GEOM + BioLIP
# reused; PLINDER protein/complex + CrossDocked retokenized with CASF held out),
# then train the ESM3-style masked-LM from scratch for up to 3 epochs. Because
# CASF is truly held out, extra epochs improve generalization instead of
# memorizing native poses (which is what sank the leaky continuation).
# gpu_1 = full H100, ~8h/epoch. Held out via -hold_jid on the 3 tokenize jobs.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

# 1) Assemble the leak-free corpus (clean + retokenized ingredients).
.venv/bin/python pipelines/corpora/mix.py \
    --inputs \
        data/lm_tokens_geom_allatom \
        data/lm_tokens_protein_plinder_nocasf \
        data/lm_tokens_complex_plinder_nocasf \
        data/lm_tokens_allatom_nocasf \
        data/lm_tokens_complex_biolip \
    --out-dir data/lm_tokens_pretrain_nocasf

# 2) Train the leak-free MLM from scratch.
.venv/bin/python pipelines/train/mlm.py \
    --token-dir data/lm_tokens_pretrain_nocasf \
    --atom-codebook-size 8192 \
    --micro-batch-size 256 --num-workers 7 \
    --max-epochs 3 --early-stop-patience 2 \
    --run-name mlm_nocasf_v1

echo "LEAK-FREE MLM DONE"

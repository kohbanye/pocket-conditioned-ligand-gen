#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -N mlm_allatom

# Production pretrain of the self-implemented ESM3-style bidirectional masked-LM
# (~99M params) over all-atom complex tokens -- the representation backbone for
# pose rescoring (a decoy pose gets a lower masked pseudo-likelihood).
#
# Corpus (~1.135B tokens, single all-atom codebook => vocab 8199, VQ xzkjxu9q):
#   data/lm_tokens_pretrain_rescore_bio = pretrain_rescore (GEOM ligand 361M +
#   PLINDER protein-only 451M + PLINDER complexes 83M + CrossDocked complexes
#   211M) + BioLIP2 complexes (30M, 128k distinct pockets, CASF/CrossDocked-test
#   excluded). Warm-started from the 1-epoch validation run (val/loss 1.13,
#   docking power 35% zero-shot) for a head start; fresh optimizer + schedule.
#
# NOTE (leakage): PLINDER complexes are NOT yet CASF-excluded -- fine for
# iterating, but the final benchmarked model needs PLINDER-complex retokenised
# with the CASF-2016 core PDBs dropped.
#
# Single GPU (gpu_1, full H100 96GB) at 2.07 it/s ~= 8h/epoch -> ~3 epochs in
# 24h (held-out-pocket val + early-stop picks the best). WANDB offline.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

.venv/bin/python scripts/train_mlm.py \
    --token-dir data/lm_tokens_pretrain_rescore_bio \
    --atom-codebook-size 8192 \
    --micro-batch-size 256 \
    --num-workers 7 \
    --max-epochs 3 \
    --early-stop-patience 2 \
    --init-from pocket-ligand-mlm/liqftueb/checkpoints/mlm-e00-vl1.1349.ckpt \
    --run-name mlm_allatom_bio_v2

echo "MLM ALLATOM PRODUCTION PRETRAIN DONE"

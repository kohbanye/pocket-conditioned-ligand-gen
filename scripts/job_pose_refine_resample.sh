#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=8:00:00
#$ -N refine_resample

# PoseBusters iteration 3: re-tokenize the legacy pose-refine set with
# CODEBOOK-RESAMPLING corruption (decode deliberately-inconsistent VQ codes ->
# realistic LM-like intramolecular errors the refiner must repair; this is a
# refiner-training augmentation, NOT a change to the deployed VQ-VAE/LM), then
# train with the known-good iter1 config (bond-graph feature, jitter 0.2,
# lambda_bond 2.0, NO angle loss -- it regressed). Target: PB-valid past 0.30
# toward the baseline band while keeping Vina. WANDB offline.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

.venv/bin/python scripts/tokenize_pose_refine.py \
    --decoder legacy \
    --ckpt "pocket-ligand-vqvae/3dvcbp0h/checkpoints/vqvae-epoch=99-val/ligand_coord=0.1501.ckpt" \
    --cache-dir data/descriptor_cache_v4 \
    --casf-pdbs data/casf2016_pdbs.txt \
    --n-complexes 8000 \
    --n-corrupt 4 \
    --resample-frac 0.3 \
    --out-dir data/pose_refine_legacy_resample

echo "TOKENIZE (resample) DONE, starting train"

.venv/bin/python scripts/train_pose_refine.py \
    --data-dir data/pose_refine_legacy_resample \
    --online-jitter-sigma 0.2 \
    --lambda-bond 2.0 \
    --micro-batch-size 16 \
    --num-workers 7 \
    --max-epochs 10 \
    --early-stop-patience 5 \
    --run-name refine_legacy_resample_v1

echo "REFINE RESAMPLE TRAIN DONE"

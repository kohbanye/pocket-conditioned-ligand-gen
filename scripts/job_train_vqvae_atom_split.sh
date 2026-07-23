#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=8:00:00
#$ -N atom_vq_split

# Retrain the all-atom VQ-VAE with SPLIT codebooks: protein atoms quantize
# against 8192 codes, ligand atoms against a dedicated 4096-code book, over ONE
# shared 33-D descriptor / encoder / decoder (routed by the source flag + a
# decoder source embedding). Fixes the shared-codebook failure where the ~10x
# more numerous protein atoms diluted ligand geometry (teacher-forced ligand
# recon connectivity collapsed 82% -> 47%; the split recovered it to 67% by
# epoch 43 already).
#
# node_f, SINGLE full H100 (--devices 1): this 5.5M VQ is communication-bound,
# so 4-GPU DDP gave ~no speedup (the first run hit the 6h wall at epoch 44). One
# full H100 (~2 min/epoch) finishes 100 epochs in ~3-4 h. batch 256 = the
# single-book baseline's batch, so the comparison isolates the split. Fresh run
# (clean LR schedule), NOT a resume of the DDP run. Checkpoints top-3 by
# val/atom_coord. WANDB offline.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline

.venv/bin/python scripts/train_vqvae_atom.py \
    --source-types cdonly \
    --cache-dir data/descriptor_cache_allatom \
    --split-codebook \
    --codebook-size 8192 \
    --ligand-codebook-size 4096 \
    --mol-batch-size 256 \
    --num-workers 8 \
    --devices 1 \
    --max-epochs 100 \
    --run-name atomvqvae-split-v2

echo "SPLIT VQ TRAINING DONE"

#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=6:00:00
#$ -N tokenize_geom
#$ -o tokenize_geom.$JOB_ID.out
#$ -e tokenize_geom.$JOB_ID.err
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$(pwd)"
module load cuda

# Stage 0: encode GEOM conformers into ligand-only LM token streams for
# pretraining. K=32 random orientations per conformer (the ligand has no pocket
# to anchor an absolute frame). Uses the SAME VQ-VAE training normalization
# stats (v4) as the CrossDocked tokenizer -- the frozen encoder must see inputs
# normalized exactly as in training or it emits garbage tokens.
#
# GEOM is streamed straight from the (single) tar -- NO extraction, so it costs
# 1 inode, not ~440k. Download it first with scripts/download_geom.py
# (Dataverse serves it decompressed; the reader auto-detects via "r|*").
#
# CPU-bound on RDKit (~100 conf/s) -> ~4-4.5 h for GEOM-drugs at max-confs 5.
# Submit: qsub -g tga-ohuelab scripts/tokenize_geom.sh
VQVAE_STATS=data/descriptor_cache_v4/normalization_stats.pt
CKPT="pocket-ligand-vqvae/3dvcbp0h/checkpoints/vqvae-epoch=99-val/ligand_coord=0.1501.ckpt"

uv run python scripts/tokenize_geom.py \
    --geom-tar data/geom/rdkit_folder.tar.gz \
    --ckpt "$CKPT" \
    --norm-stats "$VQVAE_STATS" \
    --subsets drugs \
    --max-confs-per-mol 5 \
    --num-rotations 32 \
    --ligand-codebook-size 4096 \
    --protein-codebook-size 8192 \
    --out-dir data/lm_tokens_geom \
    --batch-size 1024

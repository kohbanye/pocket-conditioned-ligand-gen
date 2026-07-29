#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=3:00:00
#$ -N stok_refine

# Pose-refiner training set for the SEPARATE-tokenizers ablation. SeparateVQVAE is
# incompatible with the pose-refiner's encode+decode round-trip, so this does NOT
# use the --separate-* flags. Instead it mirrors job_tokenize_pose_refine.sh but
# runs the all-atom decoder against the LIGAND-ONLY tokenizer (ligand-vqvae) so
# the manufactured corruption matches the separate-mode ligand VQ round-trip.
# ~12k CASF/sbdd-excluded BioLIP2 native complexes; each paired 1:1 with the
# crystal pose plus graded corruption records. Single GPU; WANDB offline.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python scripts/tokenize_pose_refine.py \
    --decoder atom \
    --ckpt pocket-ligand-vqvae/ligand-vqvae/checkpoints/last.ckpt \
    --norm-stats data/descriptor_cache_allatom/normalization_stats_ligand.pt \
    --codebook-size 8192 \
    --casf-pdbs data/casf2016_pdbs.txt \
    --n-complexes 12000 \
    --n-corrupt 4 \
    --out-dir data/pose_refine_atom_sep

echo "POSE-REFINE SEP TOKENIZE DONE"

#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=6:00:00
#$ -N stok_decoysv24096

# FAIR-ABLATION REDO (4096+4096 -> combined 8192) of job_tok_decoys_v2_sep.sh:
# the decoys_v2 pose-head corpus (12k complexes x 20 rigid+conformational
# decoys), encoded with the SEPARATE 4096 protein-VQ + 4096 ligand-VQ (unified
# into one 2*4096=8192 code space), CASF-2016 core held out.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

.venv/bin/python scripts/tokenize_decoys.py \
    --separate-protein-ckpt pocket-ligand-vqvae/protein-vqvae-4096/checkpoints/last.ckpt \
    --separate-protein-norm data/descriptor_cache_allatom/normalization_stats_protein.pt \
    --separate-ligand-ckpt pocket-ligand-vqvae/ligand-vqvae-4096/checkpoints/last.ckpt \
    --separate-ligand-norm data/descriptor_cache_allatom/normalization_stats_ligand.pt \
    --codebook-size 4096 \
    --casf-pdbs data/casf2016_pdbs.txt \
    --n-complexes 12000 --n-decoys 20 --out-dir data/lm_tokens_decoys_v2_sep4096

echo "DECOYS V2 SEP4096 TOKENIZE DONE"

#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=6:00:00
#$ -N stok_kdki

# SEPARATE-tokenizers ablation of job_tok_affinity_kdki.sh: binding-affinity
# corpus (~18k BioLIP crystal complexes carrying an experimental Kd/Ki, labelled
# with pK), encoded with the SEPARATE protein-VQ + ligand-VQ (unified into one
# code space) instead of the joint single-book atom VQ. CASF-2016 core +
# CrossDocked fold0-test held out. Crystal pose only (no decoys). node_q = 48 CPU
# (BioLIP zip streaming + per-ligand pocket carving) + 1 GPU (VQ).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
set -e

.venv/bin/python scripts/tokenize_biolip_affinity.py \
    --separate-protein-ckpt pocket-ligand-vqvae/protein-vqvae/checkpoints/last.ckpt \
    --separate-protein-norm data/descriptor_cache_allatom/normalization_stats_protein.pt \
    --separate-ligand-ckpt pocket-ligand-vqvae/ligand-vqvae/checkpoints/last.ckpt \
    --separate-ligand-norm data/descriptor_cache_allatom/normalization_stats_ligand.pt \
    --codebook-size 8192 \
    --casf-pdbs data/casf2016_pdbs.txt \
    --pk-min 2.0 --pk-max 13.0 \
    --affinity-types KD,KI \
    --out-dir data/lm_tokens_affinity_kdki_sep

echo "AFFINITY SEP TOKENIZE DONE"

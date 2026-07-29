#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=4:00:00
#$ -N stok_biolip

# SEPARATE-tokenizers ablation of job_tokenize_biolip.sh: BioLIP2 holo complexes
# tokenized into all-atom LM sequences for the rescoring pretraining corpus,
# encoded with the SEPARATE protein-VQ + ligand-VQ (unified into one code space)
# instead of the joint single-book atom VQ.
#
# CPU-BOUND (pocket + all-atom descriptor extraction per site), GPU only for the
# small VQ encode -> node_q (48 CPU, 1 GPU, coeff 0.25) with 32 workers.
# --num-rotations 1 for train (matches PLINDER complex cache). WANDB offline.
#
# Leak exclusion: CrossDocked fold0-test PDBs always; CASF-2016 core PDBs iff
# data/casf2016_pdbs.txt exists (else a warning + CASF NOT excluded).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python scripts/tokenize_biolip.py --complex \
    --separate-protein-ckpt pocket-ligand-vqvae/protein-vqvae/checkpoints/last.ckpt \
    --separate-protein-norm data/descriptor_cache_allatom/normalization_stats_protein.pt \
    --separate-ligand-ckpt pocket-ligand-vqvae/ligand-vqvae/checkpoints/last.ckpt \
    --separate-ligand-norm data/descriptor_cache_allatom/normalization_stats_ligand.pt \
    --codebook-size 8192 \
    --casf-pdbs data/casf2016_pdbs.txt \
    --num-rotations 1 \
    --num-workers 32 \
    --batch-size 512 \
    --out-dir data/lm_tokens_complex_biolip_sep

echo "BIOLIP SEP TOKENIZE DONE"

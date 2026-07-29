#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=16:00:00
#$ -N stok_geom

# SEPARATE-tokenizers ablation of the all-atom GEOM tokenization that produces
# data/lm_tokens_geom_allatom (single-range AtomLMVocab, ligand-only). Encodes
# GEOM drug conformers with the SEPARATE protein-VQ + ligand-VQ (unified into one
# code space); GEOM is ligand-only, so tokens land in the ligand half. ~1.47M
# conformers x 8 rot = ~11.6M docs. gpu_1 (cheap, coeff 0.2) with a generous 16h
# wall (descriptor recompute is single-threaded CPU; the GPU only does VQ encode).
# WANDB not used.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"

.venv/bin/python pipelines/corpora/tokenize_geom.py \
    --geom-tar data/geom/rdkit_folder.tar.gz \
    --separate-protein-ckpt pocket-ligand-vqvae/protein-vqvae/checkpoints/last.ckpt \
    --separate-protein-norm data/descriptor_cache_allatom/normalization_stats_protein.pt \
    --separate-ligand-ckpt pocket-ligand-vqvae/ligand-vqvae/checkpoints/last.ckpt \
    --separate-ligand-norm data/descriptor_cache_allatom/normalization_stats_ligand.pt \
    --codebook-size 8192 \
    --subsets drugs --max-confs-per-mol 5 --num-rotations 8 \
    --batch-size 256 \
    --out-dir data/lm_tokens_geom_allatom_sep

echo "GEOM ALLATOM SEP TOKENIZE DONE"

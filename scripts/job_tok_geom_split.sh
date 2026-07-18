#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=16:00:00
#$ -N geom_split

# B1 of the split-codebook LM pipeline: tokenize GEOM drug conformers with the
# SPLIT atom VQ (ligand atoms -> ligand book) into 2-range LMVocab ligand-only
# pretrain sequences (<bos><p></p><l> lig </l><eos>). ~1.47M conformers x 8 rot
# = ~11.6M docs. The legacy tokenize did the same conformer count in ~3h40m
# (descriptor recompute is single-threaded CPU; the GPU only does VQ encode), so
# gpu_1 (cheap, coeff 0.2) with a generous 16h wall is right. WANDB not used.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"

CKPT="pocket-ligand-vqvae/ix6q6po0/checkpoints/atomvqvae-epoch=43-val/atom_coord=0.0632.ckpt"
NORM="data/descriptor_cache_allatom/normalization_stats.pt"

.venv/bin/python scripts/tokenize_geom_atom.py \
    --geom-tar data/geom/rdkit_folder.tar.gz \
    --ckpt "$CKPT" --norm-stats "$NORM" \
    --split-codebook \
    --subsets drugs --max-confs-per-mol 5 --num-rotations 8 \
    --batch-size 256 \
    --out-dir data/lm_tokens_geom_split

echo "GEOM SPLIT TOKENIZE DONE"

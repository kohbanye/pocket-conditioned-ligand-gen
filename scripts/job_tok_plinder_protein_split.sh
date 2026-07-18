#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=4:00:00
#$ -N plprot_split

# B2 of the split-codebook LM pipeline: tokenize PLINDER pockets (protein-only,
# leakage-filtered vs CrossDocked fold0-test) with the SPLIT atom VQ (pocket
# atoms -> protein book) into 2-range LMVocab protein-only pretrain sequences
# (<bos><p> pocket </p><l></l><eos>). ~290k pockets x 8 rot = ~2.3M docs.
# node_f: 40 CPU workers stream/parse the 129 GB of zips (inode-safe, never
# extracted) + 1 GPU does VQ encode. WANDB not used.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"

CKPT="pocket-ligand-vqvae/ix6q6po0/checkpoints/atomvqvae-epoch=43-val/atom_coord=0.0632.ckpt"
NORM="data/descriptor_cache_allatom/normalization_stats.pt"

.venv/bin/python scripts/tokenize_plinder_protein.py \
    --ckpt "$CKPT" --norm-stats "$NORM" \
    --split-codebook \
    --num-rotations 8 --num-workers 40 --batch-size 256 \
    --out-dir data/lm_tokens_protein_plinder_split

echo "PLINDER PROTEIN SPLIT TOKENIZE DONE"

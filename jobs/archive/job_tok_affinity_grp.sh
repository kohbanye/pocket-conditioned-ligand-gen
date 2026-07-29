#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=6:00:00
#$ -N tok_grp

# Re-tokenize the BioLIP affinity corpus, this time emitting {split}.grp --
# one UniProt-derived protein id per doc -- so the head can be trained with a
# within-protein ranking loss.
#
# Why re-tokenize rather than patch the existing corpus: docs are dropped during
# tokenization (unparseable ligand, heavy-atom bounds, pocket setup failure), so
# the doc order cannot be reconstructed after the fact without redoing the work.
#
# Kd/Ki/IC50 all kept: the Kd/Ki-only filter was measured to halve the corpus
# (18k -> 8.6k) and cost 0.044 scoring R -- volume beat label purity.
#
# val is now split by PROTEIN, not by PDB id, so the same protein under another
# PDB id can no longer sit on both sides of the split.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
set -e

VQ="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
NORM=data/descriptor_cache_allatom/normalization_stats.pt

.venv/bin/python pipelines/corpora/tokenize_affinity_biolip.py \
    --ckpt "$VQ" \
    --norm-stats "$NORM" \
    --affinity-types KD,KI,IC50 \
    --out-dir data/lm_tokens_affinity_grp

echo "TOK AFFINITY GRP DONE"

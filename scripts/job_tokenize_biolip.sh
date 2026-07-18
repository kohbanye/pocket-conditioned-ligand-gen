#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=4:00:00
#$ -N tok_biolip

# Tokenize BioLIP2 holo complexes into all-atom LM sequences for the rescoring
# pretraining corpus. ~989k biologically-relevant sites over ~146k PDBs (vs
# CrossDocked's ~2.9k pockets) -> ~70% yield -> ~600-700k complex docs.
#
# CPU-BOUND (pocket + all-atom descriptor extraction per site), GPU only for the
# small atom-VQ encode -> node_q (48 CPU, 1 GPU, coeff 0.25) with 40 workers, not
# gpu_1 (8 CPU) with 32 workers. Single all-atom codebook (VQ xzkjxu9q) => vocab
# 8199, matching
# data/lm_tokens_pretrain_rescore. --num-rotations 2 for train (matches PLINDER
# complex cache). WANDB offline.
#
# Leak exclusion: CrossDocked fold0-test PDBs always; CASF-2016 core PDBs iff
# data/casf2016_pdbs.txt exists (else a warning + CASF NOT excluded).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python scripts/tokenize_biolip.py --complex \
    --ckpt "pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt" \
    --norm-stats data/descriptor_cache_allatom/normalization_stats.pt \
    --casf-pdbs data/casf2016_pdbs.txt \
    --num-rotations 1 \
    --num-workers 32 \
    --batch-size 512 \
    --out-dir data/lm_tokens_complex_biolip

echo "BIOLIP TOKENIZE DONE"

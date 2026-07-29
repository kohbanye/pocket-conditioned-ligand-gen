#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=4:00:00
#$ -N stok_plcplx4096

# FAIR-ABLATION REDO (4096+4096 -> combined 8192) of
# job_tok_plinder_complex_nocasf_sep.sh: PLINDER drug-like complexes
# (<p>pocket</p><l>ligand</l>) encoded with the SEPARATE 4096 protein-VQ + 4096
# ligand-VQ (unified into one 2*4096=8192 code space), CASF-2016 core held out.
# node_q = 48 CPU + 1 GPU.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
set -e

.venv/bin/python scripts/tokenize_plinder_protein.py \
    --complex \
    --separate-protein-ckpt pocket-ligand-vqvae/protein-vqvae-4096/checkpoints/last.ckpt \
    --separate-protein-norm data/descriptor_cache_allatom/normalization_stats_protein.pt \
    --separate-ligand-ckpt pocket-ligand-vqvae/ligand-vqvae-4096/checkpoints/last.ckpt \
    --separate-ligand-norm data/descriptor_cache_allatom/normalization_stats_ligand.pt \
    --codebook-size 4096 \
    --casf-pdbs data/casf2016_pdbs.txt \
    --num-rotations 2 --num-workers 40 --batch-size 256 --mw-min 150 --mw-max 600 \
    --out-dir data/lm_tokens_complex_plinder_nocasf_sep4096

echo "PLINDER COMPLEX NOCASF SEP4096 TOKENIZE DONE"

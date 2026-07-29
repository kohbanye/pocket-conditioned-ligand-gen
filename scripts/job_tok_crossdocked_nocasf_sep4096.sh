#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=4:00:00
#$ -N stok_cd4096

# FAIR-ABLATION REDO (4096+4096 -> combined 8192) of job_tok_crossdocked_nocasf_sep.sh:
# CrossDocked complexes (pocket-split, cap 128/pocket, x4 rot) encoded with the
# SEPARATE 4096 protein-VQ + 4096 ligand-VQ (unified into one 2*4096=8192 code
# space) so the separate arm's LM vocab matches the joint (8192). CASF-2016 core
# held out. node_q = 48 CPU + 1 GPU.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
set -e

.venv/bin/python scripts/tokenize_dataset_atom.py \
    --separate-protein-ckpt pocket-ligand-vqvae/protein-vqvae-4096/checkpoints/last.ckpt \
    --separate-protein-norm data/descriptor_cache_allatom/normalization_stats_protein.pt \
    --separate-ligand-ckpt pocket-ligand-vqvae/ligand-vqvae-4096/checkpoints/last.ckpt \
    --separate-ligand-norm data/descriptor_cache_allatom/normalization_stats_ligand.pt \
    --codebook-size 4096 \
    --source-types cdonly --pocket-split --max-per-pocket 128 --num-rotations 4 \
    --casf-pdbs data/casf2016_pdbs.txt \
    --out-dir data/lm_tokens_allatom_nocasf_sep4096

echo "CROSSDOCKED NOCASF SEP4096 TOKENIZE DONE"

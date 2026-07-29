#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=6:00:00
#$ -N stok_allatom4096

# FAIR-ABLATION REDO (4096+4096 -> combined 8192) of job_tok_allatom_sep.sh:
# separate-tokenizers twin of the joint data/lm_tokens_allatom (the CrossDocked
# good-pose corpus, x4 rot, default fold split, NOT CASF-held) used inside
# goodmix. Encoded with the SEPARATE 4096 protein-VQ + 4096 ligand-VQ so
# goodmix_sep4096 matches joint goodmix apples-to-apples (same complexes,
# separate 4096 tokenizer).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python pipelines/corpora/tokenize_crossdocked.py \
    --separate-protein-ckpt pocket-ligand-vqvae/protein-vqvae-4096/checkpoints/last.ckpt \
    --separate-protein-norm data/descriptor_cache_allatom/normalization_stats_protein.pt \
    --separate-ligand-ckpt pocket-ligand-vqvae/ligand-vqvae-4096/checkpoints/last.ckpt \
    --separate-ligand-norm data/descriptor_cache_allatom/normalization_stats_ligand.pt \
    --codebook-size 4096 \
    --cache-dir data/descriptor_cache_allatom \
    --out-dir data/lm_tokens_allatom_sep4096 \
    --source-types cdonly \
    --num-rotations 4 \
    --batch-size 512 \
    --splits train val

echo "ALLATOM SEP4096 TOKENIZE DONE"

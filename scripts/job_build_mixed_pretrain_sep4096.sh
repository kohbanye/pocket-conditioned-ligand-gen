#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=2:00:00
#$ -N sbuild_mix4096

# FAIR-ABLATION REDO (4096+4096 -> combined 8192) of job_build_mixed_pretrain_sep.sh:
# assemble the SEPARATE-4096-tokenizers leak-free MLM pretrain corpus by
# concatenating the five *_sep4096 token caches (geom, plinder-protein,
# plinder-complex, crossdocked-nocasf, biolip) into one mixed cache.
# build_mixed_pretrain_cache.py is a pure-CPU file concatenation (no GPU / no
# torch), so this is the cheap gpu_1 slice with a short wall -- it does NOT train
# the MLM. MUST run only after all five input corpora exist. All inputs share
# vocab 8199 (separate-mode AtomLMVocab, atom_codebook_size 2*4096=8192).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python scripts/build_mixed_pretrain_cache.py \
    --inputs \
        data/lm_tokens_geom_allatom_sep4096 \
        data/lm_tokens_protein_plinder_nocasf_sep4096 \
        data/lm_tokens_complex_plinder_nocasf_sep4096 \
        data/lm_tokens_allatom_nocasf_sep4096 \
        data/lm_tokens_complex_biolip_sep4096 \
    --out-dir data/lm_tokens_pretrain_nocasf_sep4096

echo "MIXED PRETRAIN SEP4096 CACHE DONE"

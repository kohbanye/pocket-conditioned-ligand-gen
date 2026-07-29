#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=2:00:00
#$ -N sbuild_pmix

# SEPARATE-tokenizers counterpart of the JOINT lm_tokens_pretrain_mixed corpus
# (812M tokens = protein_plinder + geom_allatom) used to pretrain the joint LM
# p6lpk7br's stage 1. Concatenate the *_sep token caches into one mixed pretrain
# cache, mirroring job_build_mixed_pretrain_sep.sh.
#   geom_allatom_sep         -- exact _sep match of the joint geom_allatom.
#   protein_plinder_nocasf_sep -- stands in for the joint's non-nocasf
#       protein_plinder (no protein_plinder_sep exists; the nocasf variant is
#       CASF-leak-free and consistent with the rest of the _sep suite).
# build_mixed_pretrain_cache.py is a pure-CPU file concatenation (no GPU / no
# torch), so this is the cheap gpu_1 slice with a short wall. MUST run only after
# both input corpora exist. All inputs share vocab 16391 (separate-mode
# AtomLMVocab, atom_codebook_size 16384).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python scripts/build_mixed_pretrain_cache.py \
    --inputs \
        data/lm_tokens_geom_allatom_sep \
        data/lm_tokens_protein_plinder_nocasf_sep \
    --out-dir data/lm_tokens_pretrain_mixed_sep

echo "BUILD PRETRAIN MIXED SEP CACHE DONE"

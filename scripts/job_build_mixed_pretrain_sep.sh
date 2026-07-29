#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=2:00:00
#$ -N sbuild_mix

# Assemble the SEPARATE-tokenizers leak-free pretrain corpus: concatenate the
# five *_sep token caches (geom, plinder-protein, plinder-complex,
# crossdocked-nocasf, biolip) into one mixed cache, mirroring the build step of
# job_build_train_mlm_nocasf.sh. build_mixed_pretrain_cache.py is a pure-CPU
# file concatenation (no GPU / no torch), so this is the cheap gpu_1 slice with a
# short wall -- it does NOT train the MLM. MUST run only after all five input
# corpora exist. All inputs share vocab 2*8192 (separate-mode AtomLMVocab).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python scripts/build_mixed_pretrain_cache.py \
    --inputs \
        data/lm_tokens_geom_allatom_sep \
        data/lm_tokens_protein_plinder_nocasf_sep \
        data/lm_tokens_complex_plinder_nocasf_sep \
        data/lm_tokens_allatom_nocasf_sep \
        data/lm_tokens_complex_biolip_sep \
    --out-dir data/lm_tokens_pretrain_nocasf_sep

echo "MIXED PRETRAIN SEP CACHE DONE"

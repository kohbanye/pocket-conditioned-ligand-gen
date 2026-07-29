#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=2:00:00
#$ -N sbuild_good4096

# FAIR-ABLATION REDO (4096+4096 -> combined 8192) of job_build_goodmix_sep.sh:
# separate-4096 counterpart of the JOINT lm_tokens_atom_goodmix corpus
# (complex_plinder_nocasf + allatom) used for the generation LM's stage 3
# placement finetune. Concatenate the *_sep4096 token caches into one mixed
# placement cache.
#   complex_plinder_nocasf_sep4096 -- _sep4096 match of the joint component
#       (PLINDER complexes, diverse pockets, CASF held out).
#   allatom_sep4096                -- CASF-held-out CrossDocked good poses (the
#       leak-free good-pose _sep4096 corpus).
# build_mixed_pretrain_cache.py is a pure-CPU file concatenation (no GPU / no
# torch), so this is the cheap gpu_1 slice with a short wall. MUST run only after
# both input corpora exist. All inputs share vocab 8199 (separate-mode
# AtomLMVocab, atom_codebook_size 2*4096=8192).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python pipelines/corpora/mix.py \
    --inputs \
        data/lm_tokens_complex_plinder_nocasf_sep4096 \
        data/lm_tokens_allatom_sep4096 \
    --out-dir data/lm_tokens_goodmix_sep4096

echo "BUILD GOODMIX SEP4096 CACHE DONE"

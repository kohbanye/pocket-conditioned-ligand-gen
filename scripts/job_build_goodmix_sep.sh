#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=2:00:00
#$ -N sbuild_good

# SEPARATE-tokenizers counterpart of the JOINT lm_tokens_atom_goodmix corpus
# (293M tokens = complex_plinder_nocasf + allatom) used for the joint LM
# p6lpk7br's stage 3 placement finetune. Concatenate the *_sep token caches into
# one mixed placement cache, mirroring job_build_mixed_pretrain_sep.sh.
#   complex_plinder_nocasf_sep -- exact _sep match of the joint component
#       (PLINDER complexes, diverse pockets, CASF held out).
#   allatom_nocasf_sep         -- CASF-held-out CrossDocked good poses. The joint
#       goodmix used the larger non-CASF-held-out `allatom` (815k good poses);
#       no CASF-held-out _sep good-pose cache of that size exists, so this is the
#       leak-free good-pose _sep corpus (173k docs). goodmix must be CASF-held
#       out, so this substitution is deliberate.
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
        data/lm_tokens_complex_plinder_nocasf_sep \
        data/lm_tokens_allatom_sep \
    --out-dir data/lm_tokens_goodmix_sep

echo "BUILD GOODMIX SEP CACHE DONE"

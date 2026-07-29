#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=2:00:00
#$ -N jbuild_good

# JOINT-tokenizer stage-3 placement corpus (293M tokens = complex_plinder_nocasf
# + allatom): concatenate the joint-tokenized token caches into one mixed
# placement cache. Composition-match of the SEPARATE arm's job_build_goodmix_sep.sh
# (complex_plinder_nocasf_sep + allatom_sep), but built from the JOINT
# single-codebook corpora (vocab 8199) so the LM ablation differs only in
# tokenizer.
#   complex_plinder_nocasf  -- PLINDER complexes, diverse pockets, CASF held out.
#   allatom                 -- CrossDocked good poses (x4 rot, 210M tokens).
# build_mixed_pretrain_cache.py is a pure-CPU file concatenation (no GPU / no
# torch), so this is the cheap gpu_1 slice with a short wall. MUST run only after
# both input corpora exist. Both inputs share vocab 8199 (joint-mode AtomLMVocab,
# atom_codebook_size 8192).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python pipelines/corpora/mix.py \
    --inputs \
        data/lm_tokens_complex_plinder_nocasf \
        data/lm_tokens_allatom \
    --out-dir data/lm_tokens_goodmix_joint

echo "BUILD GOODMIX JOINT CACHE DONE"

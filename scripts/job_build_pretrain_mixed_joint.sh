#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=2:00:00
#$ -N jbuild_pmix

# JOINT-tokenizer stage-1 pretrain corpus: concatenate the joint-tokenized
# protein_plinder_nocasf + geom_allatom token caches into one mixed pretrain
# cache. Exact composition-match of the SEPARATE arm's
# job_build_pretrain_mixed_sep.sh (geom_allatom_sep + protein_plinder_nocasf_sep),
# but built from the JOINT single-codebook corpora (vocab 8199) so the LM
# ablation differs only in tokenizer.
#   geom_allatom            -- ligand-only marginal (GEOM).
#   protein_plinder_nocasf  -- protein-only marginal (PLINDER, CASF-leak-free).
# build_mixed_pretrain_cache.py is a pure-CPU file concatenation (no GPU / no
# torch), so this is the cheap gpu_1 slice with a short wall. MUST run only after
# both input corpora exist. Both inputs share vocab 8199 (joint-mode AtomLMVocab,
# atom_codebook_size 8192).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python scripts/build_mixed_pretrain_cache.py \
    --inputs \
        data/lm_tokens_geom_allatom \
        data/lm_tokens_protein_plinder_nocasf \
    --out-dir data/lm_tokens_pretrain_mixed_joint

echo "BUILD PRETRAIN MIXED JOINT CACHE DONE"

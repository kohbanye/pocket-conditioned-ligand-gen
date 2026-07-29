#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=3:00:00
#$ -N sconcat_atomfull

# Concatenate the 10 partial separate-tokenized CrossDocked-full corpora
# (job_tok_atom_full_sep_array.sh) into the single data/lm_tokens_allatom_full_sep
# consumed by the LM full-finetune. Pure-CPU concat (build_mixed_pretrain_cache).
# Chain: qsub -hold_jid stok_atomfull_p.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python scripts/build_mixed_pretrain_cache.py \
    --inputs \
        data/lm_tokens_allatom_full_sep_p0 \
        data/lm_tokens_allatom_full_sep_p1 \
        data/lm_tokens_allatom_full_sep_p2 \
        data/lm_tokens_allatom_full_sep_p3 \
        data/lm_tokens_allatom_full_sep_p4 \
        data/lm_tokens_allatom_full_sep_p5 \
        data/lm_tokens_allatom_full_sep_p6 \
        data/lm_tokens_allatom_full_sep_p7 \
        data/lm_tokens_allatom_full_sep_p8 \
        data/lm_tokens_allatom_full_sep_p9 \
    --out-dir data/lm_tokens_allatom_full_sep

echo "ATOM_FULL SEP CONCAT DONE"

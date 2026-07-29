#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=3:00:00
#$ -N sconcat_atomfull4096

# FAIR-ABLATION REDO of job_concat_atom_full_sep.sh: concatenate the 10 partial
# separate-4096-tokenized CrossDocked-full corpora (job_tok_atom_full_sep4096_array.sh)
# into the single data/lm_tokens_allatom_full_sep4096 consumed by the LM
# full-finetune. Pure-CPU concat (build_mixed_pretrain_cache). Chain:
# qsub -hold_jid stok_atomfull4096p.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python pipelines/corpora/mix.py \
    --inputs \
        data/lm_tokens_allatom_full_sep4096_p0 \
        data/lm_tokens_allatom_full_sep4096_p1 \
        data/lm_tokens_allatom_full_sep4096_p2 \
        data/lm_tokens_allatom_full_sep4096_p3 \
        data/lm_tokens_allatom_full_sep4096_p4 \
        data/lm_tokens_allatom_full_sep4096_p5 \
        data/lm_tokens_allatom_full_sep4096_p6 \
        data/lm_tokens_allatom_full_sep4096_p7 \
        data/lm_tokens_allatom_full_sep4096_p8 \
        data/lm_tokens_allatom_full_sep4096_p9 \
    --out-dir data/lm_tokens_allatom_full_sep4096

echo "ATOM_FULL SEP4096 CONCAT DONE"

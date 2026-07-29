#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=3:00:00
#$ -N jconcat_atomfull

# Concatenate the 10 partial joint-tokenized CrossDocked-full corpora
# (job_tok_atom_full_joint_array.sh) into the single
# data/lm_tokens_allatom_full_joint consumed by the joint LM full-finetune.
# Pure-CPU concat (build_mixed_pretrain_cache). Joint-arm mirror of
# job_concat_atom_full_sep.sh. Chain: qsub -hold_jid jtok_atomfull_p.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python scripts/build_mixed_pretrain_cache.py \
    --inputs \
        data/lm_tokens_allatom_full_joint_p0 \
        data/lm_tokens_allatom_full_joint_p1 \
        data/lm_tokens_allatom_full_joint_p2 \
        data/lm_tokens_allatom_full_joint_p3 \
        data/lm_tokens_allatom_full_joint_p4 \
        data/lm_tokens_allatom_full_joint_p5 \
        data/lm_tokens_allatom_full_joint_p6 \
        data/lm_tokens_allatom_full_joint_p7 \
        data/lm_tokens_allatom_full_joint_p8 \
        data/lm_tokens_allatom_full_joint_p9 \
    --out-dir data/lm_tokens_allatom_full_joint

echo "ATOM_FULL JOINT CONCAT DONE"

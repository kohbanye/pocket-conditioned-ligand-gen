#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=6:00:00
#$ -N jtok_atomfull_p
#$ -t 1-10

# PARALLEL joint-tokenizer tokenization of the FULL CrossDocked corpus
# (2.72B tokens) split 10 ways by shard (shard_idx % 10 == partition_index).
# Exact mirror of job_tok_atom_full_sep_array.sh with only the tokenizer swapped:
# the JOINT single-codebook atom VQ (xzkjxu9q, 8192 codes, vocab 8199) replaces
# the separate protein-VQ + ligand-VQ pair. Everything else (corpus, pocket
# split, seed, rotations, batch, partitioning) is identical, so this is the
# clean tokenizer ablation. Pocket train/val split is derived from the
# manifest+seed identically in every task, so partial corpora concatenate
# without leakage. Each task -> its own _p<idx> out-dir; concatenate with
# job_concat_atom_full_joint.sh. ~2-3h/task on gpu_1 at ~the same total points.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

IDX=$((SGE_TASK_ID - 1))

.venv/bin/python scripts/tokenize_dataset_atom.py \
    --ckpt "pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt" \
    --norm-stats data/descriptor_cache_allatom/normalization_stats.pt \
    --codebook-size 8192 \
    --cache-dir data/descriptor_cache_atom_full \
    --out-dir "data/lm_tokens_allatom_full_joint_p${IDX}" \
    --source-types cdonly it0 it2_redocked \
    --pocket-split \
    --max-per-pocket 100000 \
    --pocket-val-frac 0.05 \
    --casf-pdbs data/casf2016_pdbs.txt \
    --num-rotations 1 \
    --batch-size 512 \
    --num-partitions 10 \
    --partition-index "${IDX}"

echo "ATOM_FULL JOINT TOKENIZE PARTITION ${IDX} DONE"

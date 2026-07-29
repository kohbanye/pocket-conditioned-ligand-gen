#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=6:00:00
#$ -N stok_atomfull_p
#$ -t 1-10

# PARALLEL separate-tokenizers tokenization of the FULL CrossDocked corpus
# (2.72B tokens) split 10 ways by shard (shard_idx % 10 == partition_index).
# Pocket train/val split is derived from the manifest+seed identically in every
# task, so partial corpora concatenate without leakage. Each task -> its own
# _p<idx> out-dir; concatenate with job_concat_atom_full_sep.sh. ~2-3h/task on
# gpu_1 (vs ~15-24h single) at ~the same total points.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

IDX=$((SGE_TASK_ID - 1))

.venv/bin/python pipelines/corpora/tokenize_crossdocked.py \
    --separate-protein-ckpt pocket-ligand-vqvae/protein-vqvae/checkpoints/last.ckpt \
    --separate-protein-norm data/descriptor_cache_allatom/normalization_stats_protein.pt \
    --separate-ligand-ckpt pocket-ligand-vqvae/ligand-vqvae/checkpoints/last.ckpt \
    --separate-ligand-norm data/descriptor_cache_allatom/normalization_stats_ligand.pt \
    --codebook-size 8192 \
    --cache-dir data/descriptor_cache_atom_full \
    --out-dir "data/lm_tokens_allatom_full_sep_p${IDX}" \
    --source-types cdonly it0 it2_redocked \
    --pocket-split \
    --max-per-pocket 100000 \
    --pocket-val-frac 0.05 \
    --casf-pdbs data/casf2016_pdbs.txt \
    --num-rotations 1 \
    --batch-size 512 \
    --num-partitions 10 \
    --partition-index "${IDX}"

echo "ATOM_FULL SEP TOKENIZE PARTITION ${IDX} DONE"

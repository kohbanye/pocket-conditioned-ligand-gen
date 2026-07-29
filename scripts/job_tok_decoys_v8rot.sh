#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=4:00:00
#$ -N tok_decoys_v8r

# Second pass over the SAME complexes as job_tok_decoys_v8.sh, but every copy is
# emitted under a random frame rotation (--skip-canonical). Concatenated onto v8
# it gives each complex a canonical and a rotated tokenization.
#
# Why: the pose tokens live in the pocket's PCA frame, and rotating that frame
# re-quantizes the same physical complex into almost entirely different codes
# (168/174 ligand codes change). A head fine-tuned only on canonical frames is
# measurably tied to that one tokenization -- averaging its predictions over 8
# rotations at test time DROPS docking power 89.1 -> 86.3%. Training on both
# frames should make the head score the pose rather than the code pattern.
#
#   qsub -g tga-ohuelab -t 1-12 scripts/job_tok_decoys_v8rot.sh
#   .venv/bin/python pipelines/corpora/concat_decoy_shards.py data/lm_tokens_decoys_v9

NSHARD=12

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

VQ="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"

.venv/bin/python pipelines/corpora/tokenize_decoys.py \
    --ckpt "$VQ" \
    --norm-stats data/descriptor_cache_allatom/normalization_stats.pt \
    --casf-pdbs data/casf2016_pdbs.txt \
    --n-complexes 40000 --n-decoys 24 --n-conformer-decoys 12 \
    --num-rot 1 --skip-canonical \
    --num-shards "$NSHARD" --shard-id $((SGE_TASK_ID - 1)) \
    --out-dir "data/lm_tokens_decoys_v8rot/shard$((SGE_TASK_ID - 1))"

echo "TOK DECOYS V8ROT SHARD $((SGE_TASK_ID - 1)) DONE"

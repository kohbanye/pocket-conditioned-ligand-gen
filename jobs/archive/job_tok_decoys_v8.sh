#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=4:00:00
#$ -N tok_decoys_v8

# Build the v8 pose-head corpus: 4x more complexes than v2 (which used 6.5k) and
# a THIRD decoy class -- freshly embedded ETKDG conformers rigidly superposed on
# the native pose, sampled evenly along their own RMSD order.
#
# Why: v2's decoys are all the crystal conformer, moved. A docking program's
# near-native poses are a *different* internal conformer in the right place, so
# a head trained on v2 can learn "exact crystal conformer == native" and bury a
# genuinely good redocked pose -- the diagnosed failure on 14 CASF targets where
# a 0.3-0.7 A pose ranked below 10th. The conformer class puts that band
# (best conformer ~1.3 A on average) into training.
#
# Array job, one shard of the receptor buckets per task:
#   qsub -g tga-ohuelab -t 1-12 scripts/job_tok_decoys_v8.sh
# then merge:  .venv/bin/python pipelines/corpora/concat_decoy_shards.py data/lm_tokens_decoys_v8

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
    --num-shards "$NSHARD" --shard-id $((SGE_TASK_ID - 1)) \
    --out-dir "data/lm_tokens_decoys_v8/shard$((SGE_TASK_ID - 1))"

echo "TOK DECOYS V8 SHARD $((SGE_TASK_ID - 1)) DONE"

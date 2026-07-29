#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=6:00:00
#$ -N tok_huge

# Pose corpus v10 = everything v8 has, plus the two labels/augmentations the
# head has never had:
#   * per-ligand-atom displacement (.disp/.dlen) -- dense supervision. One RMSD
#     scalar only says "this pose is wrong"; ~25 per-atom distances say WHICH
#     atoms are wrong, and their root-mean-square IS the pose label.
#   * two frame rotations per complex (--num-rot 2). A canonical-frame-only head
#     is measurably tied to one tokenization: averaging its score over 8 rotated
#     re-quantizations of the same physical pose DROPS docking power 89.1 -> 86.3%.
#
# Docs are written per complex as [native, perturbation decoys..., conformer
# decoys...] so a pose's class is recoverable from its position in the group --
# no extra sidecar needed if a class ever has to be filtered out.
#
#   qsub -g tga-ohuelab -t 1-12 scripts/job_tok_decoys_v10.sh
#   .venv/bin/python pipelines/corpora/concat_decoy_shards.py data/lm_tokens_decoys_huge

NSHARD=8

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

VQ="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"

.venv/bin/python pipelines/corpora/tokenize_decoys.py \
    --ckpt "$VQ" \
    --norm-stats data/descriptor_cache_allatom/normalization_stats.pt \
    --casf-pdbs data/casf2016_pdbs.txt \
    --seed 2 --min-heavy 70 --max-heavy 250 --n-complexes 200000 --n-decoys 16 --n-conformer-decoys 8 \
    --num-shards "$NSHARD" --shard-id $((SGE_TASK_ID - 1)) \
    --out-dir "data/lm_tokens_decoys_huge/shard$((SGE_TASK_ID - 1))"

echo "TOK DECOYS V10 SHARD $((SGE_TASK_ID - 1)) DONE"

#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=1:30:00
#$ -N eval_lf

# CASF docking-power eval of one leak-free head. $1=pooling, $2=rescore ckpt path,
# $3=output tag. Dumps per-pose scores for the consensus ensemble.
#   qsub -g tga-ohuelab scripts/job_eval_head_lf.sh meanmax <ckpt> meanmax_lf

POOL=${1:?pooling}; CKPT=${2:?ckpt}; TAG=${3:?tag}

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
set -e

.venv/bin/python scripts/eval_casf_rescore.py --score-mode head --pooling "$POOL" \
    --rescore-ckpt "$CKPT" \
    --mlm-ckpt pocket-ligand-mlm/wxlhgqx3/checkpoints/mlm-e02-vl0.8199.ckpt \
    --vqvae-ckpt "pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt" \
    --norm-stats data/descriptor_cache_allatom/normalization_stats.pt \
    --exclude-native \
    --out-csv "outputs/casf/head_${TAG}.csv" \
    --dump-scores "outputs/casf/pose_scores_${TAG}.csv"

echo "EVAL ${TAG} DONE"

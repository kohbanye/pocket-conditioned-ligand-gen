#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=1:30:00
#$ -N eval_kd2

# Recover the two KdKi cells the concurrency race left un-evaluated: mean
# (tzqaubl4) and meanmax (ynzqjqvm). Both ckpts trained fine on
# lm_tokens_affinity_kdki; only their eval was lost (the meanmax job's
# eval picked a sibling attn ckpt and crashed on the shape mismatch).
# Eval-only, so ~15 min each.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
MLM=pocket-ligand-mlm/wxlhgqx3/checkpoints/mlm-e02-vl0.8199.ckpt
VQ="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
NORM=data/descriptor_cache_allatom/normalization_stats.pt

eval_one() {
    pool=$1; ck=$2; tag=$3
    echo "=== eval $tag (pooling=$pool) ck=$ck ==="
    .venv/bin/python scripts/eval_casf_scoring.py \
        --mlm-ckpt "$MLM" --vqvae-ckpt "$VQ" --norm-stats "$NORM" \
        --rescore-ckpt "$ck" --pooling "$pool" --affinity-head \
        --out-csv "outputs/casf/affinity_${tag}.csv" || echo "FAILED $tag"
}

eval_one mean    pocket-ligand-rescore/tzqaubl4/checkpoints/rescore-e09-vl0.6196.ckpt kdki_mean
eval_one meanmax pocket-ligand-rescore/ynzqjqvm/checkpoints/rescore-e09-vl0.6159.ckpt kdki_meanmax
echo "EVAL KDKI PAIR DONE"

#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=3:00:00
#$ -N sconcat_geom4096

# Concatenate the 10 partial separate-4096-tokenized GEOM corpora
# (job_tok_geom_sep4096_array.sh) into the single data/lm_tokens_geom_allatom_sep4096
# consumed by the MLM + generation pretrain corpora. Pure-CPU concat
# (build_mixed_pretrain_cache). GEOM has train/val/test, so all three splits are
# merged. Chain: qsub -hold_jid stok_geom4096p.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

.venv/bin/python scripts/build_mixed_pretrain_cache.py \
    --splits train val test \
    --inputs \
        data/lm_tokens_geom_allatom_sep4096_p0 \
        data/lm_tokens_geom_allatom_sep4096_p1 \
        data/lm_tokens_geom_allatom_sep4096_p2 \
        data/lm_tokens_geom_allatom_sep4096_p3 \
        data/lm_tokens_geom_allatom_sep4096_p4 \
        data/lm_tokens_geom_allatom_sep4096_p5 \
        data/lm_tokens_geom_allatom_sep4096_p6 \
        data/lm_tokens_geom_allatom_sep4096_p7 \
        data/lm_tokens_geom_allatom_sep4096_p8 \
        data/lm_tokens_geom_allatom_sep4096_p9 \
    --out-dir data/lm_tokens_geom_allatom_sep4096

echo "GEOM ALLATOM SEP4096 CONCAT DONE"

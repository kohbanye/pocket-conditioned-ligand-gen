#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=12:00:00
#$ -N pretrain_lm_geom
#$ -o pretrain_lm_geom.$JOB_ID.out
#$ -e pretrain_lm_geom.$JOB_ID.err
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$(pwd)"
module load cuda

# Stage 1: pretrain the dense Qwen3 (~0.3B) LM on ligand-only GEOM tokens
# (data/lm_tokens_geom: ~1.44B train tokens, K=32) so it learns valid 3D ligand
# geometry before seeing pockets.
# node_f = 4x H100; Lightning auto-detects GPUs and uses DDP.
# ~3-5 h expected for 3 epochs (h_rt 12 h leaves margin). Save the best
# checkpoint path for the fine-tune stage (train_lm.py --init-from).
# Submit: qsub -g tga-ohuelab scripts/pretrain_lm_geom.sh
uv run python scripts/train_lm.py \
    --token-dir data/lm_tokens_geom \
    --run-name lm_geom_pretrain \
    --max-epochs 3

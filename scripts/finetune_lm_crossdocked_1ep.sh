#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=4:00:00
#$ -N finetune_lm_cd_1ep
#$ -o finetune_lm_cd_1ep.$JOB_ID.out
#$ -e finetune_lm_cd_1ep.$JOB_ID.err
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$(pwd)"
module load cuda

# Stage 2 (short): fine-tune the GEOM-pretrained LM for ONLY 1 epoch, to test
# whether a lighter touch preserves the pretrained model's better ligand
# geometry (the 3-epoch run cjp7e60q regressed to the from-scratch shape
# quality -- hypothesis: 3 epochs overwrote the pretrained geometry).
# 1 epoch ~= 1h40m on node_f; h_rt 4h leaves margin (startup + val + test).
# Submit: qsub -g tga-ohuelab scripts/finetune_lm_crossdocked_1ep.sh
PRETRAIN_CKPT="pocket-ligand-lm/gdnesyzx/checkpoints/lm-e01-vl1.8593.ckpt"

uv run python scripts/train_lm.py \
    --token-dir data/lm_tokens \
    --run-name lm_cd_finetune_1ep \
    --max-epochs 1 \
    --init-from "$PRETRAIN_CKPT"

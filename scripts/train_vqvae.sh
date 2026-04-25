#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=24:00:00
#$ -N train_vqvae
#$ -o train_vqvae.$JOB_ID.out
#$ -e train_vqvae.$JOB_ID.err
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"

module load cuda

uv run python scripts/train_vqvae.py --from-hub --hub-repo-id kohbanye/crossdocked2020 --source-types cdonly it0 it2_redocked

#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=10:00:00
#$ -N smlm4096nf

# FAIR-ABLATION REDO (separate 4096+4096 -> combined 8192) MLM backbone, NODE_F
# (4x H100 DDP) variant of job_train_mlm_sep4096.sh so it finishes in ~6-7h
# (vs ~24h on gpu_1) to meet the 12h non-gen-LM deadline. Effective batch is held
# at 256 to match the gpu_1 template + the joint/16384 MLMs (fair): micro-batch 64
# x 4 GPUs (DDP, devices="auto") = 256. Same corpus/epochs/codebook. Best by
# held-out val/loss; ckpt -> pocket-ligand-mlm/mlm_nocasf_sep4096/. Submit with
# -p -3 (max priority) -hold_jid sbuild_mix4096.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

.venv/bin/python pipelines/train/mlm.py \
    --token-dir data/lm_tokens_pretrain_nocasf_sep4096 \
    --atom-codebook-size 8192 \
    --micro-batch-size 64 --num-workers 7 \
    --max-epochs 3 --early-stop-patience 2 \
    --run-name mlm_nocasf_sep4096

echo "SEPARATE4096 MLM (node_f) DONE"

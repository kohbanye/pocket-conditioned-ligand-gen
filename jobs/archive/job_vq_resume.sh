#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=12:00:00
#$ -N vq_resume

# Resume one all-atom VQ-VAE run from its own last.ckpt on a batch node.
#
# These three runs were started interactively on r3n11 and would die with the
# session, so they move to the scheduler. trainer.fit(ckpt_path=...) restores the
# optimizer, LR schedule and epoch counter, so each picks up where it left off
# instead of restarting; if last.ckpt is missing the script fails loudly rather
# than silently burning a job on a from-scratch run.
#
# A batch node also removes the reason they were slow: on r3n11 they shared a GPU
# and a 2500-thread user limit with each other and with an unrelated tokenization
# job, which forced --num-workers 2. Alone on a gpu_1 node they get 8 cores.
#
# Usage (RUN is required; CACHE and MODALITY default to the all-atom cache):
#   qsub -g tga-ohuelab -p -3 -v RUN=ligand-vqvae-4096,MODALITY=ligand,CODEBOOK=4096 \
#        scripts/job_vq_resume.sh

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen || exit 1

# NOT sourcing ~/.bashrc: under /bin/sh it terminates the script outright (a job
# doing so exits after 0.3 s with empty output and status 0, which is
# indistinguishable from a successful no-op).
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

: "${RUN:?RUN is required (e.g. ligand-vqvae-4096)}"
: "${MODALITY:=ligand}"
: "${CODEBOOK:=4096}"
: "${CACHE:=data/descriptor_cache_allatom}"
: "${EPOCHS:=100}"

CKPT="pocket-ligand-vqvae/${RUN}/checkpoints/last.ckpt"
if [ ! -f "$CKPT" ]; then
    echo "[job] FATAL: no checkpoint to resume at $CKPT" >&2
    exit 1
fi

echo "[job] host=$(hostname) start=$(date -Is)"
echo "[job] run=$RUN modality=$MODALITY codebook=$CODEBOOK cache=$CACHE"
echo "[job] resuming from $CKPT ($(date -r "$CKPT" -Is))"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

.venv/bin/python pipelines/train/vqvae.py \
    --source-types cdonly \
    --cache-dir "$CACHE" \
    --codebook-size "$CODEBOOK" \
    --mol-batch-size 256 \
    --num-workers 8 \
    --max-epochs "$EPOCHS" \
    --modality "$MODALITY" \
    --run-name "$RUN" \
    --resume-from "$CKPT"
status=$?

echo "[job] run=$RUN exit=$status end=$(date -Is)"
exit $status

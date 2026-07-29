#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=6:00:00
#$ -N ctb_train_ref

# Train one pose-refiner recipe in the source repo, on the existing all-atom
# pose-refine set (no re-tokenisation).
#
# Target metric is vina_score, which is scored on the pose as generated. The
# measured driver of it is ligand-pocket overlap: an analytic clash relief moved
# vina_score from -2.92 to -5.59 on the 100-pocket set, while swapping between the
# three existing refiner checkpoints changed it by at most 0.2. So the recipe
# under test raises the ligand-pocket clash weight (lambda_pkt) far above the
# default, WITHOUT the rigid-body corruption that sank refine_atom_place_v2 --
# that combination has never been run.
#
# Node: gpu_1 (1 GPU, 8 CPU, 96 GB). Runtime ~3-5 h for 12-16 epochs on
#   data/pose_refine_atom (35 MB, 8000 complexes x 4 corruptions); h_rt 6 h.
#
#   qsub -g tga-ohuelab -p -3 -v RUN=refine_pkt5,LPKT=5.0,LCLASH=1.5 \
#        scripts/jobs/train_refiner.sh

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export T4TMPDIR="${T4TMPDIR:-$HOME/tmpdir}"
export TMPDIR="$T4TMPDIR"
mkdir -p "$TMPDIR"
set -e

RUN="${RUN:?set RUN}"
LPKT="${LPKT:-5.0}"
LCLASH="${LCLASH:-1.5}"
LBOND="${LBOND:-2.0}"
LANGLE="${LANGLE:-0.0}"
JITTER="${JITTER:-0.3}"
EPOCHS="${EPOCHS:-14}"
HIDDEN="${HIDDEN:-128}"
LAYERS="${LAYERS:-5}"
# e3nn tensor products are memory-hungry: hidden 192 / 6 layers OOMs a 93 GB H100
# at micro-batch 16, so raise capacity and lower the batch together.
MB="${MB:-16}"
# Training set. pose_refine_atom = VQ round-trip of crystal poses -> crystal pose.
# pose_refine_distill = LM-generated pose -> its analytically clash-relieved pose,
# i.e. the deployment distribution itself, so it needs no extra jitter.
DATA="${DATA:-data/pose_refine_atom}"
# Warm start: fine-tune an existing refiner on a new teacher instead of
# training from scratch (one epoch on the 46k-pair set costs ~1 h).
INIT="${INIT:-}"
# Rigid-body corruption teaches the refiner to RE-PLACE a mis-docked ligand,
# not just polish local geometry. The measured gap between our as-is score and
RTRANS="${RTRANS:-0}"
RROT="${RROT:-0}"
RPROB="${RPROB:-0.5}"

echo "train $RUN: data=$DATA lambda_pkt=$LPKT clash=$LCLASH bond=$LBOND angle=$LANGLE jitter=$JITTER hidden=$HIDDEN layers=$LAYERS mb=$MB epochs=$EPOCHS"

/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/.venv/bin/python pipelines/train/refiner.py \
    --data-dir "$DATA" \
    --online-jitter-sigma "$JITTER" \
    --lambda-bond "$LBOND" \
    --lambda-pkt "$LPKT" \
    --lambda-clash "$LCLASH" \
    --lambda-angle "$LANGLE" \
    --online-rigid-trans "$RTRANS" \
    --online-rigid-rot-deg "$RROT" \
    --online-rigid-prob "$RPROB" \
    --hidden-dim "$HIDDEN" \
    --n-layers "$LAYERS" \
    --micro-batch-size "$MB" \
    --num-workers 7 \
    --max-epochs "$EPOCHS" \
    --early-stop-patience 5 \
    ${INIT:+--init-from "$INIT"} \
    --run-name "$RUN"

echo "TRAIN REFINER DONE ($RUN)"
ls -1t "pocket-ligand-refine/$RUN/checkpoints/" | head -3

#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=6:00:00
#$ -N refine_geo_eval

# /loop iter2: close the gap to DiffGui (-6.54) on raw Vina.
#
# iter1 result (placement LM p6lpk7br + bond1 refiner): raw vina -4.86, already
# > DiffSBDD -4.40 / TargetDiff -4.76. Per-target: 1iep -9.11 (great), 3pbl
# -3.52, 2ity -1.95 (drags the mean; clash_free only 0.45). Two weaknesses:
#   (a) 2ity residual ligand-pocket clashes  -> raise lambda_pkt
#   (b) PB valid 0.18 everywhere (baselines 0.49-0.70) because bond1 trained
#       with lambda_angle=0 -> add bond-angle loss (PoseBusters angle check)
# Recipe = bond1's WINNING base (local jitter 0.3, lambda_bond 2.0, NO rigid
# corruption -- rigid corruption is what sank place2) + lambda_pkt 2.0 (clash)
# + lambda_clash 1.5 + lambda_angle 0.5 (geometry -> PB -> better pocket fit).
# Trains on EXISTING data/pose_refine_atom (no re-tokenize).
#
# Then generate with the placement LM and decode the SAME VQ codes three ways
# {off, geo_v1(new), bond1(prev best)} so the new refiner is A/B'd sample-for-
# sample against bond1's -4.86. Scoring: --dock-modes score min (baseline parity).

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

SC=/gs/bs/tga-ohuelab/sakano/tmp/claude-2055/-gs-bs-tga-ohuelab-sakano-git-pocket-conditioned-ligand-gen/3a2f6b7b-7888-4288-b1d5-91ee12030e9e/scratchpad
VQ="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
NORM=data/descriptor_cache_allatom/normalization_stats.pt
LM=pocket-ligand-lm/p6lpk7br/checkpoints/lm-e02-vl1.0029.ckpt
REF_BOND1=pocket-ligand-refine/refine_atom_bond_v1/checkpoints/refine-e08-r0.9440.ckpt

# --- train the geometry/clash-tuned refiner -------------------------------
.venv/bin/python scripts/train_pose_refine.py \
    --data-dir data/pose_refine_atom \
    --online-jitter-sigma 0.3 \
    --lambda-bond 2.0 \
    --lambda-pkt 2.0 \
    --lambda-clash 1.5 \
    --lambda-angle 0.5 \
    --micro-batch-size 16 \
    --num-workers 7 \
    --max-epochs 12 \
    --early-stop-patience 4 \
    --run-name refine_atom_geo_v1

echo "REFINE GEO TRAIN DONE"

REF_GEO=$(ls -1t pocket-ligand-refine/refine_atom_geo_v1/checkpoints/*.ckpt | head -1)
echo "geo refiner = $REF_GEO"

# --- generate + decode {off, geo_v1, bond1} on the SAME codes -------------
for t in 2ity 1iep 3pbl; do
    .venv/bin/python "$SC/gen_atom_target.py" "$LM" "$VQ" "$NORM" "$t" 150 \
        "${REF_GEO},${REF_BOND1}"
    OD=../sbdd-bench/outputs/own_atom/$t
    .venv/bin/python "$SC/jsonl_to_sdf.py" "$OD/generated_off.jsonl" "../sbdd-bench/outputs/own_atom_p2b_off/$t"
    .venv/bin/python "$SC/jsonl_to_sdf.py" "$OD/generated_on.jsonl"  "../sbdd-bench/outputs/own_atom_p2b_geo1/$t"
    .venv/bin/python "$SC/jsonl_to_sdf.py" "$OD/generated_on2.jsonl" "../sbdd-bench/outputs/own_atom_p2b_bond1/$t"
done

echo "GEN DONE, starting sbdd-bench evaluation"

cd /gs/bs/tga-ohuelab/sakano/git/sbdd-bench
.venv/bin/python scripts/run_evaluation.py \
    --models own_atom_p2b_off own_atom_p2b_geo1 own_atom_p2b_bond1 \
    --dock-modes score min \
    --dock-workers 7 \
    --results "$SC/results_geo1"

echo "REFINE GEO EVAL DONE"

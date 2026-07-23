#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=3:00:00
#$ -N gen_eval

# Parameterized all-atom generate + SDF export for a given (LM, refiner) pair,
# so an iteration can be evaluated the moment either model finishes training.
#
# Pass via qsub -v:
#   LM_CKPT   : LM checkpoint            (default: awdya0s8 best = full-poses model)
#   REF_CKPT  : pose refiner checkpoint  (default: newest refine_atom_place_v1)
#   TAG       : output suffix -> ../sbdd-bench/outputs/own_atom_<TAG>_{off,on}
# e.g. qsub -g tga-ohuelab -v TAG=place1 scripts/job_gen_atom_eval_param.sh

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

SC=/gs/bs/tga-ohuelab/sakano/tmp/claude-2055/-gs-bs-tga-ohuelab-sakano-git-pocket-conditioned-ligand-gen/3a2f6b7b-7888-4288-b1d5-91ee12030e9e/scratchpad
VQ="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
NORM=data/descriptor_cache_allatom/normalization_stats.pt

: "${LM_CKPT:=pocket-ligand-lm/awdya0s8/checkpoints/lm-e01-vl4.9143.ckpt}"
: "${REF_CKPT:=$(ls -1t pocket-ligand-refine/refine_atom_place_v1/checkpoints/*.ckpt 2>/dev/null | head -1)}"
: "${TAG:=place1}"

echo "LM=$LM_CKPT"
echo "REF=$REF_CKPT"
echo "TAG=$TAG"

for t in 2ity 1iep 3pbl; do
    .venv/bin/python "$SC/gen_atom_target.py" "$LM_CKPT" "$VQ" "$NORM" "$t" 150 "$REF_CKPT"
    OD=../sbdd-bench/outputs/own_atom/$t
    .venv/bin/python "$SC/jsonl_to_sdf.py" "$OD/generated_off.jsonl" "../sbdd-bench/outputs/own_atom_${TAG}_off/$t"
    .venv/bin/python "$SC/jsonl_to_sdf.py" "$OD/generated_on.jsonl"  "../sbdd-bench/outputs/own_atom_${TAG}_on/$t"
done

echo "GEN EVAL DONE TAG=$TAG"

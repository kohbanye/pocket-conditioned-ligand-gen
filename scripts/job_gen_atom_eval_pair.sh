#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=4:00:00
#$ -N gen_eval_pair

# /loop iter1 evaluation: does the PLACEMENT re-finetuned LM (p6lpk7br) fix the
# raw-Vina regression, and which refiner generation pairs best with it?
#
# One generation pass per target; the SAME VQ codes are decoded three ways, so
# every row is paired sample-for-sample:
#   *_off     : no refiner
#   *_place2  : refine_atom_place_v2 (rigid-body corruption, val rmsd 1.0095)
#   *_bond1   : refine_atom_bond_v1  (local corruption,      val rmsd 0.9440)
# Running both refiners in ONE job avoids the shared outputs/own_atom/<target>
# intermediate that would make two concurrent jobs clobber each other, and keeps
# the molecule set identical across the three arms.
#
# Baseline to beat (awdya0s8 + bond_v1): off vina_score +8.56 / on -2.14,
# PB valid 0.167 / 0.280. Target: raw vina_score ~ DiffSBDD -4.40.
# Scoring uses --dock-modes score min to match how the baseline row was scored.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

SC=/gs/bs/tga-ohuelab/sakano/tmp/claude-2055/-gs-bs-tga-ohuelab-sakano-git-pocket-conditioned-ligand-gen/3a2f6b7b-7888-4288-b1d5-91ee12030e9e/scratchpad
VQ="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
NORM=data/descriptor_cache_allatom/normalization_stats.pt

LM=pocket-ligand-lm/p6lpk7br/checkpoints/lm-e02-vl1.0029.ckpt
REF_PLACE2=pocket-ligand-refine/refine_atom_place_v2/checkpoints/refine-e09-r1.0095.ckpt
REF_BOND1=pocket-ligand-refine/refine_atom_bond_v1/checkpoints/refine-e08-r0.9440.ckpt

echo "LM=$LM"
echo "REF1=$REF_PLACE2"
echo "REF2=$REF_BOND1"

for t in 2ity 1iep 3pbl; do
    .venv/bin/python "$SC/gen_atom_target.py" "$LM" "$VQ" "$NORM" "$t" 150 \
        "${REF_PLACE2},${REF_BOND1}"
    OD=../sbdd-bench/outputs/own_atom/$t
    .venv/bin/python "$SC/jsonl_to_sdf.py" "$OD/generated_off.jsonl" "../sbdd-bench/outputs/own_atom_p2_off/$t"
    .venv/bin/python "$SC/jsonl_to_sdf.py" "$OD/generated_on.jsonl"  "../sbdd-bench/outputs/own_atom_p2_place2/$t"
    .venv/bin/python "$SC/jsonl_to_sdf.py" "$OD/generated_on2.jsonl" "../sbdd-bench/outputs/own_atom_p2_bond1/$t"
done

echo "GEN DONE, starting sbdd-bench evaluation"

cd /gs/bs/tga-ohuelab/sakano/git/sbdd-bench
.venv/bin/python scripts/run_evaluation.py \
    --models own_atom_p2_off own_atom_p2_place2 own_atom_p2_bond1 \
    --dock-modes score min \
    --dock-workers 7 \
    --results "$SC/results_place2"

echo "GEN EVAL PAIR DONE"

#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=1:00:00
#$ -N geo_eval_only

# /loop iter2 recovery: the refine_geo_eval job (8248774) trains ~28.7 min/epoch,
# so 12 epochs ends ~07:38 against an h_rt kill at 07:54 -- the trailing
# generate+evaluate stage (~16 min) gets cut. Checkpoints ARE saved per epoch,
# so this job just runs the evaluation half with the best geo_v1 checkpoint.
#
# Picks the checkpoint with the LOWEST val rmsd encoded in its filename
# (refine-eNN-rX.XXXX.ckpt) rather than the newest, so it is correct even if a
# later epoch regressed.
#
# Decodes the SAME VQ codes three ways {off, geo_v1(new), bond1(prev best -4.86)}
# so the new refiner is A/B'd sample-for-sample. Scoring: --dock-modes score min
# to match how the official baselines were scored.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

SC=/gs/bs/tga-ohuelab/sakano/tmp/claude-2055/-gs-bs-tga-ohuelab-sakano-git-pocket-conditioned-ligand-gen/3a2f6b7b-7888-4288-b1d5-91ee12030e9e/scratchpad
VQ="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
NORM=data/descriptor_cache_allatom/normalization_stats.pt
LM=pocket-ligand-lm/p6lpk7br/checkpoints/lm-e02-vl1.0029.ckpt
REF_BOND1=pocket-ligand-refine/refine_atom_bond_v1/checkpoints/refine-e08-r0.9440.ckpt

REF_GEO=$(ls -1 pocket-ligand-refine/refine_atom_geo_v1/checkpoints/*.ckpt \
    | sed 's/.*-r\([0-9.]*\)\.ckpt$/\1 &/' | sort -n | head -1 | cut -d' ' -f2)
echo "geo refiner (best by val rmsd) = $REF_GEO"
echo "bond1 refiner                  = $REF_BOND1"

for t in 2ity 1iep 3pbl; do
    .venv/bin/python "$SC/gen_atom_target.py" "$LM" "$VQ" "$NORM" "$t" 150 \
        "${REF_GEO},${REF_BOND1}"
    OD=../sbdd-bench/outputs/own_atom/$t
    .venv/bin/python "$SC/jsonl_to_sdf.py" "$OD/generated_off.jsonl" "../sbdd-bench/outputs/own_atom_p2b_off/$t"
    .venv/bin/python "$SC/jsonl_to_sdf.py" "$OD/generated_on.jsonl"  "../sbdd-bench/outputs/own_atom_p2b_geo1/$t"
    .venv/bin/python "$SC/jsonl_to_sdf.py" "$OD/generated_on2.jsonl" "../sbdd-bench/outputs/own_atom_p2b_bond1/$t"
done

echo "GEN DONE, starting sbdd-bench evaluation"

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/sbddbench
/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/.venv/bin/python scripts/run_evaluation.py \
    --models own_atom_p2b_off own_atom_p2b_geo1 own_atom_p2b_bond1 \
    --dock-modes score min \
    --dock-workers 7 \
    --results "$SC/results_geo1"

echo "GEO EVAL ONLY DONE"

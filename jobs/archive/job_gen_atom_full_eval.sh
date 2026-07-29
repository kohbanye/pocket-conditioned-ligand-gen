#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=3:00:00
#$ -N gen_atom_full

# Stage 4: generate with the CONVERGED-on-full-data all-atom LM (awdya0s8,
# best val e01) for the 3 sbdd-bench targets, decoding each sample BOTH
# refine-OFF and refine-ON with the bond-graph refiner (refine_atom_bond_v1),
# then convert to sbdd-bench generated.sdf so run_evaluation.py can score
# PoseBusters / validity / Vina.
#
# This is the decisive test of the full-data retrain: does 13.6x more ligand
# data lift PoseBusters (0.03 for the starved 8a7umbru), or is it cancelled out
# by the ~93%-decoy pose distribution that came with "all poses"?

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

SC=/gs/bs/tga-ohuelab/sakano/tmp/claude-2055/-gs-bs-tga-ohuelab-sakano-git-pocket-conditioned-ligand-gen/3a2f6b7b-7888-4288-b1d5-91ee12030e9e/scratchpad
LM=pocket-ligand-lm/awdya0s8/checkpoints/lm-e01-vl4.9143.ckpt
VQ="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
NORM=data/descriptor_cache_allatom/normalization_stats.pt
REF=pocket-ligand-refine/refine_atom_bond_v1/checkpoints/refine-e08-r0.9440.ckpt

echo "LM=$LM"
echo "refiner=$REF"

for t in 2ity 1iep 3pbl; do
    .venv/bin/python "$SC/gen_atom_target.py" "$LM" "$VQ" "$NORM" "$t" 150 "$REF"
    OD=../sbdd-bench/outputs/own_atom/$t
    .venv/bin/python "$SC/jsonl_to_sdf.py" "$OD/generated_off.jsonl" "../sbdd-bench/outputs/own_atom_full_off/$t"
    .venv/bin/python "$SC/jsonl_to_sdf.py" "$OD/generated_on.jsonl"  "../sbdd-bench/outputs/own_atom_full_on/$t"
done

echo "GEN ATOM FULL DONE"

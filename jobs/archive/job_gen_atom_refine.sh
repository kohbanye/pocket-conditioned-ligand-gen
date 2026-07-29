#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=3:00:00
#$ -N gen_atom_refine

# All-atom validation: generate poses for the 3 sbdd-bench targets with the
# all-atom LM (8a7umbru) + all-atom VQ-VAE (xzkjxu9q), decode each sample BOTH
# refine-OFF and refine-ON (same VQ codes) using the trained all-atom refiner,
# then dock both with Vina (score_as_is) -> paired refiner-effect comparison.
# NOTE: the all-atom LM is undertrained (val ~4.3), so molecule quality is low;
# the paired on/off delta isolates the REFINER's effect regardless. WANDB off.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

SC=/gs/bs/tga-ohuelab/sakano/tmp/claude-2055/-gs-bs-tga-ohuelab-sakano-git-pocket-conditioned-ligand-gen/3a2f6b7b-7888-4288-b1d5-91ee12030e9e/scratchpad
LM=pocket-ligand-lm/8a7umbru/checkpoints/lm-e01-vl4.2823.ckpt
VQ="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
NORM=data/descriptor_cache_allatom/normalization_stats.pt
CK=$(ls -1t pocket-ligand-refine/refine_atom_v1/checkpoints/*.ckpt | head -1)
echo "refiner checkpoint: $CK"

for t in 2ity 1iep 3pbl; do
    .venv/bin/python "$SC/gen_atom_target.py" "$LM" "$VQ" "$NORM" "$t" 150 "$CK"
    OD=../sbdd-bench/outputs/own_atom/$t
    RCP=../sbdd-bench/data/targets/$t/${t}_receptor.pdbqt
    .venv/bin/python scripts/dock_vina.py --jsonl "$OD/generated_off.jsonl" --receptor-pdbqt "$RCP" --out-csv "$OD/dock_off.csv" --workers 7 --limit 80
    .venv/bin/python scripts/dock_vina.py --jsonl "$OD/generated_on.jsonl"  --receptor-pdbqt "$RCP" --out-csv "$OD/dock_on.csv"  --workers 7 --limit 80
done

echo "GEN ATOM REFINE DONE"

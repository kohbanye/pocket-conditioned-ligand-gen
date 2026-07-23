#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=2:00:00
#$ -N temp_sweep

# /loop iter3: sampling-temperature sweep -- the cheapest untested Vina lever.
#
# iter2 established that the REFINER lever is exhausted for Vina (geo_v1 lifted
# PB 0.158->0.222 but Vina went -4.84 -> -4.72). Progress must come from the
# generation side. Temperature has never been tuned: every run so far used
# temperature=1.0 / top_p=0.95.
#
# Rationale: div_uniqueness is 0.99 and scaffold diversity 0.85 -- we have
# diversity to spare. iter1 per-target showed 2ity at -1.95 vs 1iep -9.11, i.e.
# the mean is dragged by badly-placed samples, exactly what a lower temperature
# suppresses. Trading surplus diversity for placement quality should raise mean
# Vina. Cost: NO training, generation only.
#
# Arms: temperature 0.7 / 0.85 / 1.0 (1.0 re-run in-job as the control so all
# three are scored under one protocol). Refiner fixed to bond1 (iter1+iter2 best
# on Vina). Each arm writes its own outputs/own_atom_t<T>/ intermediate so arms
# never clobber each other. Scoring: --dock-modes score min (baseline parity).
#
# Watch for: diversity collapse. If div_uniqueness/scaffold_diversity fall a lot
# at low T, the Vina gain is not free and must be reported as a trade-off.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

SC=/gs/bs/tga-ohuelab/sakano/tmp/claude-2055/-gs-bs-tga-ohuelab-sakano-git-pocket-conditioned-ligand-gen/3a2f6b7b-7888-4288-b1d5-91ee12030e9e/scratchpad
VQ="pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
NORM=data/descriptor_cache_allatom/normalization_stats.pt
LM=pocket-ligand-lm/p6lpk7br/checkpoints/lm-e02-vl1.0029.ckpt
REF_BOND1=pocket-ligand-refine/refine_atom_bond_v1/checkpoints/refine-e08-r0.9440.ckpt

MODELS=""
for T in 0.7 0.85 1.0; do
    TAG=$(echo "$T" | tr -d '.')          # 0.7 -> 07
    export GEN_TEMPERATURE="$T"
    export GEN_OUT_SUFFIX="_t${TAG}"
    echo "=== temperature $T (suffix $GEN_OUT_SUFFIX) ==="
    for t in 2ity 1iep 3pbl; do
        .venv/bin/python "$SC/gen_atom_target.py" "$LM" "$VQ" "$NORM" "$t" 150 "$REF_BOND1"
        OD=../sbdd-bench/outputs/own_atom_t${TAG}/$t
        .venv/bin/python "$SC/jsonl_to_sdf.py" "$OD/generated_on.jsonl" \
            "../sbdd-bench/outputs/own_t${TAG}_on/$t"
    done
    MODELS="$MODELS own_t${TAG}_on"
done

echo "GEN DONE, models:$MODELS"

cd /gs/bs/tga-ohuelab/sakano/git/sbdd-bench
.venv/bin/python scripts/run_evaluation.py \
    --models $MODELS \
    --dock-modes score min \
    --dock-workers 7 \
    --results "$SC/results_tsweep"

echo "TEMP SWEEP DONE"

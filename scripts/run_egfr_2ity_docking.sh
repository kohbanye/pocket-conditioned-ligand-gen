#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=1:30:00
#$ -N egfr_2ity_dock
#
# End-to-end EGFR 2ITY experiment: generate ~10k ligands conditioned on the
# 2ITY pocket, then dock each with Vina (as-is score_only + optimized
# local_only). node_q gives 1 GPU (for generation) + 48 CPU cores (for docking).
#
# Submit from the project root:
#     qsub -g <your-group> scripts/run_egfr_2ity_docking.sh
#
# The target must already be prepared (needs internet to download the PDB, so
# run this once on a login / interactive node first):
#     PYTHONPATH=$PWD .venv/bin/python scripts/prepare_target.py \
#         --pdb-id 2ITY --ligand-resname IRE --chain A \
#         --out-dir data/targets/2ity --tag 2ity

set -eu

source "$HOME/.bashrc"
module load cuda 2>/dev/null || true

ROOT=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
cd "$ROOT"
export PYTHONPATH=$ROOT
PY=$ROOT/.venv/bin/python

TARGET=data/targets/2ity
OUT=outputs/egfr_2ity
LM_CKPT=pocket-ligand-lm/g79let5b/checkpoints/lm-e09-vl1.4088.ckpt
NUM_SAMPLES=10000

if [ ! -f "$TARGET/2ity_receptor.pdbqt" ]; then
    echo "ERROR: target not prepared. Run scripts/prepare_target.py first." >&2
    exit 1
fi

echo "===== [1/2] Generating $NUM_SAMPLES ligands (GPU) ====="
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/generate_ligands_for_target.py \
    --receptor "$TARGET/2ity_receptor.pdb" \
    --ref-ligand "$TARGET/2ity_ref_ligand.sdf" \
    --lm-ckpt "$LM_CKPT" \
    --num-samples "$NUM_SAMPLES" --batch-size 200 --seed 0 \
    --out-dir "$OUT"

echo "===== [2/2] Docking with Vina (48 CPU cores) ====="
"$PY" scripts/dock_vina.py \
    --jsonl "$OUT/generated.jsonl" \
    --receptor-pdbqt "$TARGET/2ity_receptor.pdbqt" \
    --out-csv "$OUT/docking_results.csv" \
    --workers 48 --box-size 22.5 \
    --tmp-dir "${T4TMPDIR:-$HOME/tmpdir}"

echo "===== Done. Results in $OUT ====="

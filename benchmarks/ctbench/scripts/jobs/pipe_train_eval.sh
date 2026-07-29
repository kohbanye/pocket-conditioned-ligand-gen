#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=6:00:00
#$ -N ctb_pipe_te

# Stage 2 of the unattended refiner pipeline: train a distilled refiner on the
# CrossDocked train-pocket set, freeze the checkpoint, apply it to the evaluation
# arm and score vina_score + PoseBusters. Runs end to end with no supervision.
#
# The checkpoint is copied to pocket-ligand-refine/frozen/ before evaluation
# because Lightning keeps only the top-K files — evaluating the live path races
# with training and fails with FileNotFoundError.
#
#   qsub -g tga-ohuelab -p -3 -hold_jid <build_id> \
#     -v RUN=tp400_b10,LBOND=10,LANGLE=1,EPOCHS=12 scripts/jobs/pipe_train_eval.sh

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/ctbench
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/ctbench:/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export T4TMPDIR="${T4TMPDIR:-$HOME/tmpdir}"; export TMPDIR="$T4TMPDIR"; mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

SRC=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
SB=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/sbddbench
CT=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/benchmarks/ctbench
RUN="${RUN:?set RUN}"
DATA="${DATA:-data/pose_refine_tp400}"
LBOND="${LBOND:-2.0}"
LANGLE="${LANGLE:-0.0}"
EPOCHS="${EPOCHS:-12}"
ARM="${ARM:-sep4096_cs800}"
IDFILE="${IDFILE:-data/target_ids_ab.txt}"
INIT="${INIT:-$SRC/pocket-ligand-refine/refine_atom_bond_v1/checkpoints/refine-e08-r0.9440.ckpt}"

# ---- train ---------------------------------------------------------------
cd "$SRC"
/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/.venv/bin/python pipelines/train/refiner.py \
    --data-dir "$DATA" --init-from "$INIT" \
    --online-jitter-sigma 0 --lambda-bond "$LBOND" --lambda-angle "$LANGLE" \
    --lambda-pkt 1.0 --lambda-clash 1.0 \
    --micro-batch-size 16 --num-workers 7 \
    --max-epochs "$EPOCHS" --early-stop-patience 5 --run-name "$RUN"

mkdir -p "$SRC/pocket-ligand-refine/frozen"
BEST=$(ls -1t "$SRC/pocket-ligand-refine/$RUN/checkpoints/"*.ckpt | head -1)
cp "$BEST" "$SRC/pocket-ligand-refine/frozen/$RUN.ckpt"
echo "TRAIN DONE ($RUN) best=$(basename "$BEST")"

# ---- apply + score, at 1 and 3 network iterations -------------------------
cd "$CT"
for REP in 1 3; do
  OUT="P_${RUN}_r${REP}"
  rm -rf "$SB/outputs/$OUT"
  # shellcheck disable=SC2046
  "$SRC/.venv/bin/python" scripts/apply_refiner_to_arm.py \
      --arm "$ARM" --out-arm "$OUT" \
      --ckpt "pocket-ligand-refine/frozen/$RUN.ckpt" --repeat "$REP" \
      --targets $(tr '\n' ' ' < "$IDFILE")
  RES="$CT/results/generation/$OUT"
  mkdir -p "$RES"
  cd "$SB"
  # shellcheck disable=SC2046
  /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/.venv/bin/python scripts/run_evaluation.py --models own \
      --index "$SB/data/targets/index.json" \
      --out-dir "$SB/outputs/$OUT" --results "$RES" \
      --ids $(tr '\n' ' ' < "$CT/$IDFILE") \
      --dock-modes score --dock-workers 7
  cd "$CT"
  echo "EVAL DONE $OUT"
done

# ---- summary -------------------------------------------------------------
"$CT/.venv/bin/python" - <<PY
import pandas as pd, os
ids=set(open("$CT/$IDFILE").read().split())
print("=== $RUN : leak-free ML-only, arm $ARM ===")
for rep in (1,3):
    p=f"$CT/results/generation/P_${RUN}_r{rep}/per_molecule.parquet"
    if not os.path.exists(p): print(f"  x{rep}: missing"); continue
    d=pd.read_parquet(p); d=d[d.get("tag","")!="ref"]; d=d[d.target_id.isin(ids)]
    pb=pd.to_numeric(d.pb_valid,errors="coerce")
    met = "  *** BOTH TARGETS MET ***" if (d.vina_score.mean()<-5.58 and pb.mean()>0.5) else ""
    print(f"  x{rep}: vina_score={d.vina_score.mean():.3f}  pb_valid={pb.mean():.3f}{met}")
PY
echo "PIPE TRAIN+EVAL DONE ($RUN)"

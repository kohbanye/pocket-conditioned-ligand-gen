#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=8:00:00
#$ -N boltz_casf
#$ -t 1-10

# Boltz-2 affinity on the CASF-2016 core (285 targets), as an array job.
# Each task takes every 10th target (stride = NTASKS), so the ~28 large proteins
# (>1000 residues) spread evenly instead of piling into one task.
#
# Boltz-2 is run in its native mode: protein sequence + ligand SMILES -> it
# predicts the structure AND the affinity. Unlike our head / GenScore / Vina it
# does NOT see the CASF crystal structure, so it is reported as a reference
# column, not a same-input comparison.
#
# Pilot: 1a30 took 6.5 min (incl. MSA fetch) => ~31 GPU-h total => ~3.2 h/task.
# Output: outputs/boltz_casf/predict/boltz_results_<pdbid>/predictions/<pdbid>/
#         affinity_<pdbid>.json  ("affinity_pred_value" = log10(IC50 [uM]); pK = 6 - y)

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PATH="$HOME/usr/app/babel/bin:$HOME/.local/bin:$PATH"
export RCSBROOT="$HOME/.boltzina/maxit-v11.300-prod-src"
export PATH="$RCSBROOT/bin:$PATH"
export BOLTZ_CACHE=/gs/bs/tga-ohuelab/sakano/.cache/boltz

BOLTZ_REPO=/gs/bs/tga-ohuelab/sakano/git/boltzina_repos_v2
YAML_DIR=outputs/boltz_casf/yaml
OUT_DIR=outputs/boltz_casf/predict
NTASKS=10
mkdir -p "$OUT_DIR" outputs/boltz_casf/logs

i=0
for y in $(ls "$YAML_DIR"/*.yaml | sort); do
    i=$((i + 1))
    # stride assignment: task t handles i where i % NTASKS == (t-1)
    if [ $(( (i - 1) % NTASKS )) -ne $(( SGE_TASK_ID - 1 )) ]; then
        continue
    fi
    pdbid=$(basename "$y" .yaml)
    if [ -f "$OUT_DIR/boltz_results_${pdbid}/predictions/${pdbid}/affinity_${pdbid}.json" ]; then
        echo "[task $SGE_TASK_ID] $pdbid already done, skipping"
        continue
    fi
    echo "[task $SGE_TASK_ID] $pdbid start $(date +%H:%M:%S)"
    (cd "$BOLTZ_REPO" && uv run boltz predict \
        "/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/$y" \
        --out_dir "/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/$OUT_DIR" \
        --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 \
        --use_msa_server) \
        > "outputs/boltz_casf/logs/${pdbid}.log" 2>&1
    echo "[task $SGE_TASK_ID] $pdbid rc=$? $(date +%H:%M:%S)"
done

echo "BOLTZ CASF TASK $SGE_TASK_ID DONE"

#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=4:00:00
#$ -N ctbench_gen_delta

# Generate ONLY the 14 multi-pair CrossDocked pockets (the 7 dirs holding 2
# receptor+ligand pairs) for one variant, to complete the 100-pocket set without
# regenerating the 86 single-pair outputs. VARIANT=joint_nocasf|separate_4096.
cd /gs/bs/tga-ohuelab/sakano/git/complex-tokenizer-bench
export CTBENCH_SOURCE_REPO=/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export CTBENCH_SBDD_PYTHON=/gs/bs/tga-ohuelab/sakano/git/sbdd-bench/.venv/bin/python
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/complex-tokenizer-bench:/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e
VARIANT="${VARIANT:-joint_nocasf}"
.venv/bin/python scripts/infer_generation_crossdocked.py --variant "$VARIANT" --n-samples 100 --skip-eval \
    --ids CDK6_HUMAN_1_312_0__2f2c_B_rec CDK6_HUMAN_1_312_0__4aua_A_rec CHIB_SERMA_1_499_0__1h0i_A_rec CHIB_SERMA_1_499_0__4z2g_A_rec LMBL1_HUMAN_198_526_0__2pqw_A_rec LMBL1_HUMAN_198_526_0__2rhy_A_rec NOS1_HUMAN_302_723_0__3tym_A_rec NOS1_HUMAN_302_723_0__4d7o_A_rec NOS3_HUMAN_65_480_0__1rs9_A_rec NOS3_HUMAN_65_480_0__4kcq_A_rec NQO1_HUMAN_2_274_0__1dxo_C_rec NQO1_HUMAN_2_274_0__1gg5_A_rec PYRD_TRYCC_2_314_catalytic_0__2e6d_A_rec PYRD_TRYCC_2_314_catalytic_0__3w83_B_rec
echo "CROSSDOCKED DELTA GEN DONE ($VARIANT)"

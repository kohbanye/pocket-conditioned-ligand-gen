#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=12:00:00
#$ -N stok_geom4096p
#$ -t 1-10

# FAIR-ABLATION REDO (4096+4096 -> combined 8192), PARALLEL geom tokenize.
# SEPARATE 4096 protein-VQ + 4096 ligand-VQ (GEOM is ligand-only, so tokens land
# in the ligand half of the 2*4096=8192 combined AtomLMVocab). ~1.47M conformers
# x 8 rot split 10 ways by conformer-stream index (conf_idx % 10 == partition).
# The molecule->split assignment is derived from SMILES+seed identically in every
# task, so partial caches concatenate without leakage. Each task -> its own
# _p<idx> out-dir; concatenate with job_concat_geom_sep4096.sh.
#
# NOTE: unlike the atom_full array (seekable .pt shards it can skip), the GEOM
# gzip/tar is streamed sequentially, so every partition re-streams + re-parses the
# whole archive; only the (bottleneck) descriptor recompute + rotations + VQ
# encode are partitioned. Wall is generous (12h) to guard against silent
# truncation. SUBMIT p1 FIRST (-t 1-1) and check its ETA before launching the
# rest; if the per-task rate approaches the single-job time, use the single-GPU
# fallback job_tok_geom_sep4096.sh (16h) instead.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"
export WANDB_MODE=offline
set -e

IDX=$((SGE_TASK_ID - 1))

.venv/bin/python scripts/tokenize_geom_atom.py \
    --geom-tar data/geom/rdkit_folder.tar.gz \
    --separate-protein-ckpt pocket-ligand-vqvae/protein-vqvae-4096/checkpoints/last.ckpt \
    --separate-protein-norm data/descriptor_cache_allatom/normalization_stats_protein.pt \
    --separate-ligand-ckpt pocket-ligand-vqvae/ligand-vqvae-4096/checkpoints/last.ckpt \
    --separate-ligand-norm data/descriptor_cache_allatom/normalization_stats_ligand.pt \
    --codebook-size 4096 \
    --subsets drugs --max-confs-per-mol 5 --num-rotations 8 \
    --batch-size 256 \
    --num-partitions 10 \
    --partition-index "${IDX}" \
    --out-dir "data/lm_tokens_geom_allatom_sep4096_p${IDX}"

echo "GEOM ALLATOM SEP4096 TOKENIZE PARTITION ${IDX} DONE"

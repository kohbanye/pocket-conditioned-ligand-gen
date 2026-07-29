#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=16:00:00
#$ -N stok_geom4096

# FALLBACK (single-GPU) FAIR-ABLATION REDO (4096+4096 -> combined 8192) of
# job_tok_geom_sep.sh. Use this INSTEAD of job_tok_geom_sep4096_array.sh +
# job_concat_geom_sep4096.sh when partitioning does not pay off (the GEOM gzip/tar
# is streamed sequentially, so 10 partitions re-parse the whole archive; if that
# dominates, a single 16h job is cheaper and safer). Do NOT run both paths -- they
# write the same out-dir. SEPARATE 4096 protein-VQ + 4096 ligand-VQ; GEOM is
# ligand-only so tokens land in the ligand half of the 2*4096=8192 AtomLMVocab.
# gpu_1 (coeff 0.2), 16h wall (descriptor recompute is single-threaded CPU; GPU
# only does VQ encode). WANDB not used.

cd /gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen
export PYTHONPATH="/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen:${PYTHONPATH}"

.venv/bin/python pipelines/corpora/tokenize_geom.py \
    --geom-tar data/geom/rdkit_folder.tar.gz \
    --separate-protein-ckpt pocket-ligand-vqvae/protein-vqvae-4096/checkpoints/last.ckpt \
    --separate-protein-norm data/descriptor_cache_allatom/normalization_stats_protein.pt \
    --separate-ligand-ckpt pocket-ligand-vqvae/ligand-vqvae-4096/checkpoints/last.ckpt \
    --separate-ligand-norm data/descriptor_cache_allatom/normalization_stats_ligand.pt \
    --codebook-size 4096 \
    --subsets drugs --max-confs-per-mol 5 --num-rotations 8 \
    --batch-size 256 \
    --out-dir data/lm_tokens_geom_allatom_sep4096

echo "GEOM ALLATOM SEP4096 TOKENIZE DONE"

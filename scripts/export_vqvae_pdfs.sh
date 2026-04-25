#!/bin/sh
# Export one PDF per VQ-VAE run by driving notebooks/visualization.py with
# a different VQVAE_CKPT each time.  Runs the notebook end-to-end (inference
# on 2k test complexes + 200-complex 3D RMSD + t-SNE) so each export needs a
# GPU; sequence them so they don't fight over the device.
#
# Usage: ./scripts/export_vqvae_pdfs.sh
#
# Edit the CKPTS map below to add/remove runs.
set -e

NB=notebooks/visualization.py

export_one() {
    tag=$1
    ckpt=$2
    out=notebooks/visualization_${tag}.pdf
    echo "=== Exporting ${tag} → ${out} ==="
    VQVAE_CKPT="${ckpt}" uv run marimo export pdf "${NB}" -o "${out}" -f
}

export_one baseline \
    "/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/pocket-ligand-vqvae/yggua4f0/checkpoints/vqvae-epoch=99-val/protein_recon=0.1316.ckpt"
export_one b1a \
    "/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/pocket-ligand-vqvae/lv5nldy5/checkpoints/vqvae-epoch=96-val/protein_recon=0.1294.ckpt"
export_one b1b \
    "/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/pocket-ligand-vqvae/uhhyc6y7/checkpoints/vqvae-epoch=84-val/protein_recon=0.1286.ckpt"
export_one b1c \
    "/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/pocket-ligand-vqvae/t65o9cot/checkpoints/vqvae-epoch=99-val/protein_recon=0.1571.ckpt"

echo "All exports complete."

"""Fetch / link model weights and evaluation data.

- FoldToken4: download model_zoom.zip from Zenodo into weights/foldtoken/.
- ESM3: trigger the HuggingFace download of the structure tokenizer weights
  (requires `huggingface-cli login` and license acceptance for the gated repo).
- Own VQ-VAE: symlink the trained checkpoint + descriptor cache from the
  separate working copy into weights/ and data/ (source stays where it is).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from plbench import paths  # noqa: E402

FOLDTOKEN_ZENODO = "https://zenodo.org/records/13901445/files/model_zoom.zip?download=1"
# Default source checkpoint in the separate working copy (override with --own-src).
# 3dvcbp0h matches the current model code (descriptor_dim 65/30) and was trained
# on descriptor_cache_v4, whose normalization stats must be linked alongside.
OWN_CKPT_SRC = (
    paths.OWN_MODEL_WORKDIR
    / "pocket-ligand-vqvae/3dvcbp0h/checkpoints/vqvae-epoch=99-val/ligand_coord=0.1501.ckpt"
)
OWN_CACHE_SRC = paths.OWN_MODEL_WORKDIR / "data/descriptor_cache_v4"


def fetch_foldtoken() -> None:
    dest = paths.WEIGHTS_DIR / "foldtoken"
    dest.mkdir(parents=True, exist_ok=True)
    if paths.FOLDTOKEN_CKPT.exists():
        print(f"[foldtoken] already present: {paths.FOLDTOKEN_CKPT}")
        return
    zip_path = dest / "model_zoom.zip"
    print(f"[foldtoken] downloading {FOLDTOKEN_ZENODO} ...")
    urllib.request.urlretrieve(FOLDTOKEN_ZENODO, zip_path)  # noqa: S310
    print("[foldtoken] extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    zip_path.unlink(missing_ok=True)
    print(f"[foldtoken] done -> {paths.FOLDTOKEN_FT4_DIR}")


def fetch_esm3() -> None:
    # Public, non-gated repo. Only the structure encoder/decoder are needed for
    # reconstruction (~1.3 GB), not the full 5.5 GB model.
    print(f"[esm3] downloading structure tokenizer weights ({paths.ESM3_HF_REPO}) ...")
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=paths.ESM3_HF_REPO,
            allow_patterns=[
                "config.json",
                "data/*.json",
                "data/weights/esm3_structure_encoder_v0.pth",
                "data/weights/esm3_structure_decoder_v0.pth",
            ],
        )
        print(f"[esm3] cached at {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[esm3] FAILED: {exc}")


def link_own(own_src: Path, cache_src: Path) -> None:
    ckpt_dst = paths.OWN_VQVAE_CKPT
    ckpt_dst.parent.mkdir(parents=True, exist_ok=True)
    if not own_src.exists():
        print(f"[own] source checkpoint not found: {own_src}")
    else:
        _symlink(own_src, ckpt_dst)
        print(f"[own] {ckpt_dst} -> {own_src}")
    if cache_src.exists():
        _symlink(cache_src, paths.OWN_DESCRIPTOR_CACHE)
        print(f"[own] {paths.OWN_DESCRIPTOR_CACHE} -> {cache_src}")
    else:
        print(f"[own] descriptor cache not found: {cache_src}")


def _symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--foldtoken", action="store_true")
    p.add_argument("--esm3", action="store_true")
    p.add_argument("--own", action="store_true")
    p.add_argument("--own-src", type=Path, default=OWN_CKPT_SRC)
    p.add_argument("--own-cache-src", type=Path, default=OWN_CACHE_SRC)
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    paths.ensure_dirs()
    if args.all or args.foldtoken:
        fetch_foldtoken()
    if args.all or args.esm3:
        fetch_esm3()
    if args.all or args.own:
        link_own(args.own_src, args.own_cache_src)
    if not any([args.all, args.foldtoken, args.esm3, args.own]):
        p.print_help()


if __name__ == "__main__":
    main()

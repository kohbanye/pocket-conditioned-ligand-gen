"""Fetch / link model weights and evaluation data.

- FoldToken4: download model_zoom.zip from Zenodo into weights/foldtoken/.
- ESM3: trigger the HuggingFace download of the structure tokenizer weights
  (requires `huggingface-cli login` and license acceptance for the gated repo).

ProLIT needs nothing here: its adapter reads each arm's checkpoint straight out
of the run directories under ``paths.OWN_VQ_RUNS_DIR``.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from recon_bench import paths  # noqa: E402

FOLDTOKEN_ZENODO = "https://zenodo.org/records/13901445/files/model_zoom.zip?download=1"


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
    _patch_foldtoken_config()
    print(f"[foldtoken] done -> {paths.FOLDTOKEN_FT4_DIR}")


def _patch_foldtoken_config() -> None:
    """The shipped FT4 config omits ``k_neighbors`` (a required model arg, 30
    everywhere in the code). Add it so reconstruct.py can build the model."""
    cfg = paths.FOLDTOKEN_CONFIG
    if cfg.exists() and "k_neighbors" not in cfg.read_text():
        with cfg.open("a") as f:
            f.write("k_neighbors: 30\n")
        print(f"[foldtoken] patched k_neighbors into {cfg.name}")


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




def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--foldtoken", action="store_true")
    p.add_argument("--esm3", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    paths.ensure_dirs()
    if args.all or args.foldtoken:
        fetch_foldtoken()
    if args.all or args.esm3:
        fetch_esm3()
    if not any([args.all, args.foldtoken, args.esm3]):
        p.print_help()


if __name__ == "__main__":
    main()

"""Link / locate model weights.

* **Ours** (``--own``): symlink the trained LM + VQ-VAE checkpoints and the
  descriptor cache (normalization stats) from the separate working copy into
  ``weights/own/`` and ``data/`` — source stays where it is.
* **DiffSBDD** (``--diffsbdd``): symlink the CrossDocked conditional checkpoint
  (already present in the local DiffSBDD working copy), else print the source.
* **TargetDiff** / **DiffGui** (``--targetdiff`` / ``--diffgui``): these ship
  their checkpoints on Google Drive; we link a local copy if found and otherwise
  print exactly where to put the file. ``gdown`` is used if available.

Every external checkpoint path is overridable via the env vars in
``sbddbench/paths.py`` (e.g. ``SBDD_TARGETDIFF_CKPT``).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sbddbench import paths  # noqa: E402

# Known local copies on this machine (best-effort; overridden by --*-src).
LOCAL_DIFFSBDD_CKPT = Path(
    os.environ.get("SBDD_DIFFSBDD_CKPT_SRC", "")
    or str(paths.WEIGHTS_DIR / "diffsbdd" / "crossdocked_fullatom_cond.ckpt")
)

# Upstream sources (for the printed instructions).
SRC = {
    "targetdiff": "https://drive.google.com/drive/folders/1-ftaIrTXjWFhw3-0Twkrs5m0yX6CNarz "
                  "(pretrained_diffusion.pt) — TargetDiff README §'Trained model checkpoint'",
    "diffgui": "https://drive.google.com/drive/folders/1pQk1FASCnCLjYRd7yc17WfctoHR50s2r "
               "(trained.pt + bond_trained.pt) — DiffGui README §'Trained model checkpoint'",
}


def _symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())
    print(f"  linked {dst}  ->  {src}")


def link_own(lm_src: Path, vqvae_src: Path, cache_src: Path) -> None:
    print("[own]")
    for src, dst, what in [
        (lm_src, paths.OWN_LM_CKPT, "LM checkpoint"),
        (vqvae_src, paths.OWN_VQVAE_CKPT, "VQ-VAE checkpoint"),
        (cache_src, paths.OWN_DESCRIPTOR_CACHE, "descriptor cache"),
    ]:
        if Path(src).exists():
            _symlink(Path(src), dst)
        else:
            print(f"  MISSING {what}: {src}")


def link_diffsbdd(src: Path) -> None:
    print("[diffsbdd]")
    if Path(src).exists():
        _symlink(Path(src), paths.DIFFSBDD_CKPT)
    else:
        print(f"  MISSING checkpoint: {src}\n"
              "  Download crossdocked_fullatom_cond.ckpt from the DiffSBDD repo "
              "(Zenodo, README) and place it at "
              f"{paths.DIFFSBDD_CKPT} (or set SBDD_DIFFSBDD_CKPT).")


def _try_gdown(folder_url: str, dest_dir: Path) -> bool:
    if shutil.which("gdown") is None:
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    import subprocess

    r = subprocess.run(["gdown", "--folder", folder_url.split()[0], "-O", str(dest_dir)],
                       capture_output=True, text=True)
    return r.returncode == 0


def locate_external(name: str, dst: Path, also: list[Path] | None = None) -> None:
    print(f"[{name}]")
    targets = [dst, *(also or [])]
    missing = [p for p in targets if not p.exists()]
    if not missing:
        for p in targets:
            print(f"  present: {p}")
        return
    for p in missing:
        print(f"  MISSING: {p}")
    print(f"  Source: {SRC[name]}")
    print(f"  Place the file(s) at the path(s) above (or set the SBDD_{name.upper()}_CKPT env var).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--own", action="store_true")
    p.add_argument("--diffsbdd", action="store_true")
    p.add_argument("--targetdiff", action="store_true")
    p.add_argument("--diffgui", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--own-lm-src", type=Path, default=paths.OWN_LM_CKPT_SRC)
    p.add_argument("--own-vqvae-src", type=Path, default=paths.OWN_VQVAE_CKPT_SRC)
    p.add_argument("--own-cache-src", type=Path, default=paths.OWN_DESCRIPTOR_CACHE_SRC)
    p.add_argument("--diffsbdd-src", type=Path, default=LOCAL_DIFFSBDD_CKPT)
    args = p.parse_args()

    paths.ensure_dirs()
    if args.all or args.own:
        link_own(args.own_lm_src, args.own_vqvae_src, args.own_cache_src)
    if args.all or args.diffsbdd:
        link_diffsbdd(args.diffsbdd_src)
    if args.all or args.targetdiff:
        locate_external("targetdiff", paths.TARGETDIFF_CKPT)
    if args.all or args.diffgui:
        locate_external("diffgui", paths.DIFFGUI_CKPT,
                        also=[paths.DIFFGUI_CKPT.parent / "bond_trained.pt"])
    if not any([args.all, args.own, args.diffsbdd, args.targetdiff, args.diffgui]):
        p.print_help()


if __name__ == "__main__":
    main()

"""Fetch baseline weights that are not carried by the submodules.

- OnionNet-2: upstream tracks ``62shell_saved-model.h5`` with git-lfs, so a
  submodule checkout on a machine without git-lfs has no model at all (or a
  ~130-byte pointer). It is fetched from Zenodo instead.

GIGN needs nothing here: its three checkpoints are ordinary files in its
repository. GenScore's checkpoints likewise ship in ``trained_models/``.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from affinity_bench import paths  # noqa: E402

#: An HDF5 file starts with this signature.
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
#: Zenodo record 10728089 holds exactly one file, ~884.6 MB.
EXPECTED_BYTES = 884_639_216


def fetch_onionnet2(*, force: bool = False) -> None:
    dest = paths.ONIONNET2_MODEL
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force and _looks_like_hdf5(dest):
        print(f"[onionnet2] already present: {dest} ({dest.stat().st_size} bytes)")
        return
    print(f"[onionnet2] downloading {paths.ONIONNET2_ZENODO} (~885 MB) ...")
    urllib.request.urlretrieve(paths.ONIONNET2_ZENODO, dest)  # noqa: S310
    size = dest.stat().st_size
    if not _looks_like_hdf5(dest):
        msg = (
            f"{dest} is not an HDF5 file after download ({size} bytes). "
            "A captive portal or an error page is the usual cause."
        )
        raise ValueError(msg)
    if size != EXPECTED_BYTES:
        print(f"[onionnet2] WARNING: got {size} bytes, expected {EXPECTED_BYTES}")
    print(f"[onionnet2] done -> {dest} ({size} bytes)")


def _looks_like_hdf5(path: Path) -> bool:
    with path.open("rb") as fh:
        return fh.read(len(HDF5_MAGIC)) == HDF5_MAGIC


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onionnet2", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.onionnet2 or args.all:
        fetch_onionnet2(force=args.force)
    elif not any((args.onionnet2, args.all)):
        ap.error("nothing selected; pass --onionnet2 or --all")


if __name__ == "__main__":
    main()

"""Download the GEOM dataset from Harvard Dataverse.

GEOM (Axelrod & Gomez-Bombarelli 2022) lives at Harvard Dataverse under
``doi:10.7910/DVN/JNGTDF``. This helper lists the dataset's files and streams
the requested one to disk via the Dataverse access API (stdlib only, no extra
deps).

Run on a login / data-transfer node (needs internet + ~8 GB free); write to
Lustre ``data/`` (NOT ``$HOME`` -- it is inode/space limited). The default
``rdkit_folder.tar.gz`` is the per-molecule RDKit-pickle distribution consumed
by :mod:`prolit.data.geom`.

    # see what's available
    uv run python pipelines/corpora/download/geom.py --list
    # fetch + extract the rdkit_folder distribution
    uv run python pipelines/corpora/download/geom.py --file rdkit_folder.tar.gz \
        --out-dir data/geom
    tar -xf data/geom/rdkit_folder.tar.gz -C data/geom
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATAVERSE = "https://dataverse.harvard.edu"
PERSISTENT_ID = "doi:10.7910/DVN/JNGTDF"
# Dataverse / its CDN returns 403 to the default ``Python-urllib`` User-Agent;
# a browser-like UA is accepted.
_UA = "Mozilla/5.0 (X11; Linux x86_64) geom-download/1.0"


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": _UA})  # noqa: S310


def list_files() -> list[dict]:
    url = (
        f"{DATAVERSE}/api/datasets/:persistentId/"
        f"?persistentId={PERSISTENT_ID}"
    )
    with urllib.request.urlopen(_request(url)) as resp:  # noqa: S310
        payload = json.loads(resp.read())
    return payload["data"]["latestVersion"]["files"]


def _download(file_id: int, dest: Path, expected_size: int | None) -> None:
    url = f"{DATAVERSE}/api/access/datafile/{file_id}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading datafile %d -> %s", file_id, dest)
    with urllib.request.urlopen(_request(url)) as resp, dest.open("wb") as out:  # noqa: S310
        downloaded = 0
        next_log = 0
        log_every = 512 << 20  # log every 512 MB
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_log:
                next_log += log_every
                if expected_size:
                    pct = 100 * downloaded / expected_size
                    logger.info(
                        "  %.1f%% (%d / %d bytes)", pct, downloaded, expected_size
                    )
                else:
                    logger.info("  %d bytes", downloaded)
    logger.info("Saved %s (%d bytes)", dest, dest.stat().st_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List files and exit.")
    parser.add_argument("--file", type=str, default="rdkit_folder.tar.gz")
    parser.add_argument("--out-dir", type=Path, default=Path("data/geom"))
    args = parser.parse_args()

    files = list_files()
    if args.list:
        for f in files:
            df = f["dataFile"]
            logger.info(
                "%-40s %12d bytes  id=%s",
                df["filename"],
                df.get("filesize", -1),
                df["id"],
            )
        return

    match = next(
        (f for f in files if f["dataFile"]["filename"] == args.file), None
    )
    if match is None:
        names = ", ".join(f["dataFile"]["filename"] for f in files)
        msg = f"File {args.file!r} not found. Available: {names}"
        raise SystemExit(msg)

    df = match["dataFile"]
    _download(df["id"], args.out_dir / args.file, df.get("filesize"))


if __name__ == "__main__":
    main()

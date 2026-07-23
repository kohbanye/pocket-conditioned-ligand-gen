# ruff: noqa: T201  (progress is printed to stdout by design)
"""Download BioLIP2 receptor + ligand structure buckets and annotations.

BioLIP2 (zhanggroup.org/BioLiP, the ``/BioLiP2/`` path 404s -- it *is* v2,
updated weekly) serves structures as ~989 receptor + ~989 ligand ``.tar.bz2``
buckets keyed by the middle two characters of the 4-char PDB id, under
``weekly/``. Together they are the complete dataset (~19 GB, ~2.6M interaction
sites over ~200k PDBs -- far more distinct pockets than PLINDER or CrossDocked).

Cloudflare gotcha: the default ``urllib`` User-Agent (and browser-emulating
fetchers) get 403; a plain browser UA header sails through with 200. So this
downloader sets a browser ``User-Agent`` -- the PLINDER pattern
(``urlretrieve`` with the default UA) does NOT work against this host.

Inode-safe: keeps every ``.tar.bz2`` as-is and NEVER extracts (the tokenizer
streams each bucket in memory). Resumable: a bucket already present is skipped
(atomic ``.part`` -> rename guarantees a present file is complete). Store on
Lustre (``data/`` under the project), not home.

Run (from the project root, on a node with internet, e.g. r3n11)::

    .venv/bin/python scripts/download_biolip.py [--workers 8] [--annotations-only]
"""

from __future__ import annotations

import argparse
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://zhanggroup.org/BioLiP/"
OUT_DIR = Path("data/biolip")
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
# Annotation + ligand-template files (relative to BASE).
ANNOTATION_FILES = [
    "download/BioLiP.txt.gz",
    "download/BioLiP_nr.txt.gz",
    "data/ligand.tsv.gz",
]
BUCKET_RE = re.compile(r"(receptor|ligand)_[0-9a-z]{2}\.tar\.bz2")


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})  # noqa: S310
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        return resp.read()


def _download_one(url: str, tgt: Path) -> tuple[str, str, int]:
    """Stream ``url`` to ``tgt`` (atomic, size-checked). Skip if present."""
    if tgt.exists() and tgt.stat().st_size > 0:
        return tgt.name, "skip", tgt.stat().st_size
    part = tgt.with_name(tgt.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": UA})  # noqa: S310
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310
                expected = int(resp.headers.get("Content-Length", 0))
                with part.open("wb") as f:
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
            if expected == 0 or part.stat().st_size == expected:
                part.rename(tgt)
                return tgt.name, "ok", tgt.stat().st_size
        except Exception:  # noqa: BLE001
            # Cloudflare 503-throttles bursts; back off exponentially (cap 90s).
            time.sleep(min(90, 5 * 2**attempt))
    if part.exists():
        part.unlink()
    return tgt.name, "FAIL", 0


def _bucket_names() -> list[str]:
    html = _get(BASE + "weekly.html").decode("utf-8", "ignore")
    return sorted({m.group(0) for m in BUCKET_RE.finditer(html)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Fetch only the annotation/ligand tables, skip the ~19 GB buckets.",
    )
    args = parser.parse_args()

    (OUT_DIR / "receptor").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ligand").mkdir(parents=True, exist_ok=True)

    # 1) annotations + ligand SMILES template table (small).
    for rel in ANNOTATION_FILES:
        name, status, sz = _download_one(BASE + rel, OUT_DIR / Path(rel).name)
        print(f"annotation {name}: {status} ({sz / 1e6:.1f} MB)", flush=True)

    if args.annotations_only:
        print("DONE (annotations only)", flush=True)
        return

    # 2) structure buckets (receptor_XX.tar.bz2 / ligand_XX.tar.bz2 under weekly/).
    buckets = _bucket_names()
    jobs: list[tuple[str, Path]] = []
    for name in buckets:
        kind = "receptor" if name.startswith("receptor") else "ligand"
        jobs.append((BASE + "weekly/" + name, OUT_DIR / kind / name))
    print(f"{len(jobs)} buckets to fetch (~19 GB)", flush=True)

    done = n_ok = n_skip = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_download_one, url, tgt) for url, tgt in jobs]
        for i, fut in enumerate(as_completed(futs)):
            name, status, sz = fut.result()
            if status == "ok":
                n_ok += 1
                done += sz
            elif status == "skip":
                n_skip += 1
                done += sz
            else:
                n_fail += 1
                print("FAIL", name, flush=True)
            if (i + 1) % 50 == 0:
                print(
                    f"{i + 1}/{len(jobs)} | {done / 1e9:.1f} GB"
                    f" | ok {n_ok} skip {n_skip} fail {n_fail}",
                    flush=True,
                )

    print(f"DONE: ok {n_ok} skip {n_skip} fail {n_fail}", flush=True)


if __name__ == "__main__":
    main()

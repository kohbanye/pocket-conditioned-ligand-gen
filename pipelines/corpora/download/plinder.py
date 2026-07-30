
"""Download PLINDER systems/*.zip from the public GCS bucket.

Inode-safe: keeps the ~1,060 zip shards as-is and NEVER extracts them (the
downstream pocket tokenizer streams each zip in memory). Resumable: a shard
already present with the correct size is skipped. Atomic: writes ``.part`` then
renames.

Run (from the project root, on a node with internet, e.g. r3n11)::

    .venv/bin/python pipelines/corpora/download/plinder.py [--workers 8]
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Object names from the GCS listing exclude the bucket, so the bucket name
# ``plinder`` must be part of the base URL.
BASE = "https://storage.googleapis.com/plinder/"
LIST_PATH = Path("data/plinder/systems_list.txt")
OUT_DIR = Path("data/plinder/systems")


def _parse_list() -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    for line in LIST_PATH.read_text().splitlines():
        if not line.strip():
            continue
        name, size = line.split("\t")
        items.append((name, int(size)))
    return items


def _download_one(name: str, size: int) -> tuple[str, str, int]:
    tgt = OUT_DIR / Path(name).name
    if tgt.exists() and tgt.stat().st_size == size:
        return name, "skip", size
    part = tgt.with_name(tgt.name + ".part")
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(BASE + name, str(part))  # noqa: S310
            if part.stat().st_size == size:
                part.rename(tgt)
                return name, "ok", size
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    if part.exists():
        part.unlink()
    return name, "FAIL", 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = _parse_list()
    total = sum(s for _, s in items)
    done = 0
    n_ok = n_skip = n_fail = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_download_one, n, s) for n, s in items]
        for i, fut in enumerate(as_completed(futs)):
            name, status, sz = fut.result()
            if status in ("ok", "empty"):
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
                    f"{i + 1}/{len(items)} | {done / 1e9:.1f}/{total / 1e9:.1f} GB"
                    f" | ok {n_ok} skip {n_skip} fail {n_fail}",
                    flush=True,
                )

    print(f"DONE: ok {n_ok} skip {n_skip} fail {n_fail}", flush=True)


if __name__ == "__main__":
    main()

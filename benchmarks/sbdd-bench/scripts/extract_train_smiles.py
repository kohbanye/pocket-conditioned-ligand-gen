"""Canonical SMILES of every ligand in the training corpus, for novelty.

``run_evaluation.py`` has taken ``--train-smiles`` all along, but no file was
ever built, so Novelty has never been reported. It is one of the standard
columns in this benchmark's table, and without it "the model generates new
molecules" is an assertion rather than a measurement.

Reads the ligand files the descriptor cache was built from, so the set matches
what the model actually saw. They are extracted ``*.sdf.gz`` rather than tars
in this checkout under ``hub_cache/ligands``; the full set is the tars under
``hub_cache/repo/ligands``. Both are handled: ``*.tar`` are streamed, loose
``*.sdf.gz`` are read directly. The generation-benchmark pockets are excluded from
training, but their *ligands* can appear with other pockets -- that is exactly
what novelty is meant to expose, so nothing is filtered here.
"""

from __future__ import annotations

import argparse
import gzip
import tarfile
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-tars", type=int, default=None,
                   help="cap the number of ligand files (smoke runs)")
    a = p.parse_args()

    seen: set[str] = set()

    def add(text: str) -> None:
        # one mol is enough: every pose in a file is the same ligand
        mol = Chem.MolFromMolBlock(text.split("$$$$")[0], sanitize=True)
        if mol is not None:
            seen.add(Chem.MolToSmiles(mol))

    tars = sorted(a.repo_dir.rglob("*.tar"))
    if a.max_tars:
        tars = tars[: a.max_tars]
    if tars:
        print(f"{len(tars)} tars", flush=True)
        for ti, tp in enumerate(tars):
            n = 0
            try:
                with tarfile.open(tp, "r|") as tf:
                    for m in tf:
                        if not m.name.endswith(".sdf.gz"):
                            continue
                        fh = tf.extractfile(m)
                        if fh is None:
                            continue
                        n += 1
                        try:
                            add(gzip.decompress(fh.read()).decode("utf-8", "replace"))
                        except (OSError, EOFError):
                            continue
            except (tarfile.TarError, OSError) as exc:
                print(f"  skip {tp.name}: {exc!r}", flush=True)
            print(f"  tar {ti + 1}/{len(tars)}  files {n}  smiles {len(seen)}",
                  flush=True)
    else:
        loose = sorted(a.repo_dir.rglob("*.sdf.gz"))
        print(f"{len(loose)} loose ligand files", flush=True)
        for i, fp in enumerate(loose):
            try:
                add(gzip.decompress(fp.read_bytes()).decode("utf-8", "replace"))
            except (OSError, EOFError):
                continue
            if (i + 1) % 20000 == 0:
                print(f"  {i + 1}/{len(loose)}  smiles {len(seen)}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(sorted(seen)))
    print(f"wrote {len(seen)} canonical SMILES to {a.out}")


if __name__ == "__main__":
    main()

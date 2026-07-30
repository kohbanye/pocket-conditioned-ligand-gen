"""Dock generated ligands against a prepared receptor with AutoDock Vina.

Reads ``generated.jsonl`` (see ``generate_ligands_for_target.py``) and, for
each dockable ligand, runs two Vina evaluations against the receptor:

* **as-is** (``--score_only``): score the generated pose exactly as produced.
* **optimized** (``--local_only``): local minimisation starting from the
  generated pose, then score. Also reports the heavy-atom RMSD between the
  generated and minimised poses (how far minimisation had to move it).

This is the standard SBDD docking protocol (cf. TargetDiff's Vina Score /
Vina Min). Ligand prep: Open Babel perceives bonds from the generated 3D
coordinates, protonates at pH 7.4, assigns Gasteiger charges, and writes a
flexible ``.pdbqt`` (the generated heavy-atom coordinates are preserved).

The docking box is a cube of ``--box-size`` Å centred on each ligand's own
centroid: this always contains the ligand (Vina scores are accurate as long
as the ligand is inside the box, since receptor grids are built from the full
receptor), so there are no "ligand outside the search space" failures.

Parallelised across CPU cores (each Vina call is single-threaded). Writes
``docking_results.csv`` incrementally.

Example::

    PYTHONPATH=$PWD .venv/bin/python scripts/dock_vina.py \
        --jsonl outputs/egfr_2ity/generated.jsonl \
        --receptor-pdbqt data/targets/2ity/2ity_receptor.pdbqt \
        --out-csv outputs/egfr_2ity/docking_results.csv --workers 48
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

from prolit.external_tools import tool_default

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_VINA = tool_default("vina")
DEFAULT_OBABEL = tool_default("obabel")

_SCORE_RE = re.compile(
    r"Estimated Free Energy of Binding\s*:\s*(-?\d+\.?\d*)", re.IGNORECASE
)

# Worker globals (set by _init_worker; avoids pickling per task).
_CFG: dict = {}


def _init_worker(cfg: dict) -> None:
    global _CFG  # noqa: PLW0603
    _CFG = cfg


def _write_xyz(path: Path, elements: list[str], coords: list[list[float]]) -> None:
    lines = [str(len(elements)), "generated"]
    lines += [
        f"{el} {x:.4f} {y:.4f} {z:.4f}"
        for el, (x, y, z) in zip(elements, coords, strict=True)
    ]
    path.write_text("\n".join(lines) + "\n")


def _read_pdbqt_heavy(path: Path) -> list[tuple[float, float, float]]:
    """Heavy-atom coordinates from a pdbqt (AutoDock H types start with 'H')."""
    out: list[tuple[float, float, float]] = []
    for ln in path.read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        atype = ln[77:79].strip() or ln.split()[-1]
        if atype.upper().startswith("H"):
            continue
        out.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    return out


def _heavy_rmsd(
    a: list[tuple[float, float, float]], b: list[tuple[float, float, float]]
) -> float | None:
    if not a or len(a) != len(b):
        return None
    sq = sum(
        (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2
        for (ax, ay, az), (bx, by, bz) in zip(a, b, strict=True)
    )
    return (sq / len(a)) ** 0.5


def _run(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        cmd, capture_output=True, text=True, check=False, timeout=timeout
    )


def _parse_score(stdout: str) -> float | None:
    m = _SCORE_RE.search(stdout)
    return float(m.group(1)) if m else None


def dock_one(record: dict) -> dict:
    cfg = _CFG
    idx = record["idx"]
    tag = record["tag"]
    elements = record["elements"]
    coords = record["coords"]
    row: dict = {
        "idx": idx, "tag": tag, "n_atoms": len(elements), "n_atoms_docked": None,
        "dock_ok": False, "score_as_is": None, "score_opt": None,
        "opt_rmsd": None, "reason": "",
    }
    n = len(elements)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]
    cx, cy, cz = sum(xs) / n, sum(ys) / n, sum(zs) / n
    # Box centred on the ligand, each edge large enough to contain the ligand
    # (its extent + margin) with a floor of --box-size. Guarantees the ligand
    # is inside the search space; Vina scores are accurate as long as it is.
    margin = 8.0
    min_size = cfg["box_size"]
    sx = max(min_size, (max(xs) - min(xs)) + margin)
    sy = max(min_size, (max(ys) - min(ys)) + margin)
    sz = max(min_size, (max(zs) - min(zs)) + margin)
    box = [
        "--center_x", f"{cx:.3f}", "--center_y", f"{cy:.3f}", "--center_z", f"{cz:.3f}",
        "--size_x", f"{sx:.3f}", "--size_y", f"{sy:.3f}", "--size_z", f"{sz:.3f}",
    ]
    common = ["--receptor", cfg["receptor"], "--cpu", "1", "--seed", "1", *box]

    with tempfile.TemporaryDirectory(dir=cfg["tmp_dir"]) as td:
        tdp = Path(td)
        xyz = tdp / "lig.xyz"
        pdbqt = tdp / "lig.pdbqt"
        _write_xyz(xyz, elements, coords)
        # -r keeps only the largest contiguous fragment: generated point clouds
        # sometimes break into pieces under bond perception, which would emit a
        # multi-ROOT pdbqt that Vina rejects. Docking the largest fragment is the
        # standard SBDD fallback.
        ob = _run([
            cfg["obabel"], str(xyz), "-O", str(pdbqt), "-r",
            "-p", "7.4", "--partialcharge", "gasteiger",
        ])
        if not pdbqt.exists() or not any(
            ln.startswith(("ATOM", "HETATM")) for ln in pdbqt.read_text().splitlines()
        ):
            row["reason"] = f"obabel failed: {ob.stderr.strip()[:120]}"
            return row
        row["n_atoms_docked"] = len(_read_pdbqt_heavy(pdbqt))
        try:
            r1 = _run([cfg["vina"], "--ligand", str(pdbqt), "--score_only", *common])
            row["score_as_is"] = _parse_score(r1.stdout)
            optp = tdp / "opt.pdbqt"
            r2 = _run([
                cfg["vina"], "--ligand", str(pdbqt), "--local_only",
                "--out", str(optp), *common,
            ])
            row["score_opt"] = _parse_score(r2.stdout)
            if optp.exists():
                row["opt_rmsd"] = _heavy_rmsd(
                    _read_pdbqt_heavy(pdbqt), _read_pdbqt_heavy(optp)
                )
        except subprocess.TimeoutExpired:
            row["reason"] = "vina timeout"
            return row
    row["dock_ok"] = row["score_as_is"] is not None or row["score_opt"] is not None
    if not row["dock_ok"]:
        row["reason"] = row["reason"] or "vina produced no score"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--receptor-pdbqt", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--box-size", type=float, default=22.5)
    parser.add_argument(
        "--limit", type=int, default=None, help="Dock only first N (smoke test)."
    )
    parser.add_argument("--vina", type=str, default=DEFAULT_VINA)
    parser.add_argument("--obabel", type=str, default=DEFAULT_OBABEL)
    parser.add_argument(
        "--tmp-dir", type=str, default=os.environ.get("T4TMPDIR")
        or str(Path.home() / "tmpdir"),
    )
    args = parser.parse_args()

    import json  # noqa: PLC0415

    records = [
        json.loads(ln) for ln in args.jsonl.read_text().splitlines() if ln.strip()
    ]
    dockable = [r for r in records if r.get("dockable")]
    # Always keep the reference control (idx=-1) if present and dockable.
    if args.limit is not None:
        ref = [r for r in dockable if r["idx"] < 0]
        rest = [r for r in dockable if r["idx"] >= 0][: args.limit]
        dockable = ref + rest
    logger.info(
        "%d records, %d dockable -> docking with %d workers (box %.1f A)",
        len(records), len(dockable), args.workers, args.box_size,
    )

    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "receptor": str(args.receptor_pdbqt),
        "vina": args.vina, "obabel": args.obabel,
        "box_size": args.box_size, "tmp_dir": args.tmp_dir,
    }

    fields = [
        "idx", "tag", "n_atoms", "n_atoms_docked", "dock_ok",
        "score_as_is", "score_opt", "opt_rmsd", "reason",
    ]
    n_done = 0
    n_ok = 0
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        with Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for row in pool.imap_unordered(dock_one, dockable, chunksize=4):
                writer.writerow(row)
                n_done += 1
                n_ok += int(row["dock_ok"])
                if n_done % 200 == 0:
                    f.flush()
                    logger.info("docked %d/%d (ok %d)", n_done, len(dockable), n_ok)

    logger.info("\nDone. Docked %d ligands (%d ok) -> %s", n_done, n_ok, args.out_csv)


if __name__ == "__main__":
    main()

"""Re-dock the top-scoring generated ligands and save their minimised poses.

The batch docker (``dock_vina.py``) discards the optimized pose to save space.
This script takes the best ligands by Vina Min (optimized / ``local_only``)
score, re-runs the local minimisation while keeping the output pose, and writes
a receptor+ligand complex PDB per hit (plus a ligand-only SDF) so the docked
structures can be inspected in PyMOL / a viewer.

Example::

    PYTHONPATH=$PWD .venv/bin/python scripts/extract_top_poses.py \
        --jsonl outputs/egfr_2ity/generated.jsonl \
        --dock-csv outputs/egfr_2ity/docking_results.csv \
        --receptor-pdbqt data/targets/2ity/2ity_receptor.pdbqt \
        --receptor-pdb data/targets/2ity/2ity_receptor.pdb \
        --out-dir outputs/egfr_2ity/top_poses --top 10
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.write_reconstruction_pdbs import (  # noqa: E402
    infer_bonds,
    write_full_protein_pdb,
)
from src.tokenizers.ligand import parse_sdf  # noqa: E402

DEFAULT_VINA = "/home/5/uq02055/.local/bin/vina"
DEFAULT_OBABEL = "/home/5/uq02055/usr/app/babel/bin/obabel"
_SCORE_RE = re.compile(r"Estimated Free Energy of Binding\s*:\s*(-?\d+\.?\d*)")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--dock-csv", type=Path, required=True)
    parser.add_argument("--receptor-pdbqt", type=Path, required=True)
    parser.add_argument("--receptor-pdb", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--box-size", type=float, default=22.5)
    parser.add_argument("--vina", type=str, default=DEFAULT_VINA)
    parser.add_argument("--obabel", type=str, default=DEFAULT_OBABEL)
    parser.add_argument("--tmp-dir", type=str, default=str(Path.home() / "tmpdir"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)

    coords_by_idx = {}
    for ln in args.jsonl.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            coords_by_idx[r["idx"]] = r

    dock = list(csv.DictReader(args.dock_csv.open()))
    scored = [
        r for r in dock
        if int(r["idx"]) >= 0 and r["dock_ok"] == "True" and r["score_opt"]
    ]
    scored.sort(key=lambda r: float(r["score_opt"]))
    top = scored[: args.top]
    logger.info("Top %d by Vina Min; writing complexes to %s", len(top), args.out_dir)

    summary = []
    for rank, row in enumerate(top, start=1):
        idx = int(row["idx"])
        rec = coords_by_idx[idx]
        elements = rec["elements"]
        coords = rec["coords"]
        n = len(coords)
        cx = sum(c[0] for c in coords) / n
        cy = sum(c[1] for c in coords) / n
        cz = sum(c[2] for c in coords) / n
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]
        margin = 8.0
        sx = max(args.box_size, (max(xs) - min(xs)) + margin)
        sy = max(args.box_size, (max(ys) - min(ys)) + margin)
        sz = max(args.box_size, (max(zs) - min(zs)) + margin)

        with tempfile.TemporaryDirectory(dir=args.tmp_dir) as td:
            tdp = Path(td)
            xyz = tdp / "lig.xyz"
            xyz.write_text(
                f"{n}\ngen_{idx}\n"
                + "\n".join(
                    f"{e} {x:.4f} {y:.4f} {z:.4f}"
                    for e, (x, y, z) in zip(elements, coords, strict=True)
                )
                + "\n"
            )
            pdbqt = tdp / "lig.pdbqt"
            _run([
                args.obabel, str(xyz), "-O", str(pdbqt), "-r",
                "-p", "7.4", "--partialcharge", "gasteiger",
            ])
            opt = tdp / "opt.pdbqt"
            res = _run([
                args.vina, "--receptor", str(args.receptor_pdbqt),
                "--ligand", str(pdbqt), "--local_only", "--out", str(opt),
                "--cpu", "1", "--seed", "1",
                "--center_x", f"{cx:.3f}", "--center_y", f"{cy:.3f}",
                "--center_z", f"{cz:.3f}",
                "--size_x", f"{sx:.3f}", "--size_y", f"{sy:.3f}",
                "--size_z", f"{sz:.3f}",
            ])
            m = _SCORE_RE.search(res.stdout)
            score = float(m.group(1)) if m else float(row["score_opt"])
            # Optimized pose -> SDF (elements + bonds), then heavy atoms.
            lig_sdf = tdp / "opt.sdf"
            _run([args.obabel, str(opt), "-O", str(lig_sdf)])
            mol = parse_sdf(lig_sdf)[0]
            heavy = [(a[0], a[1], a[2], a[3]) for a in mol["atoms"] if a[0] != "H"]
            elems = [h[0] for h in heavy]
            import numpy as np  # noqa: PLC0415

            xyz_arr = np.array([[h[1], h[2], h[3]] for h in heavy], dtype=np.float64)
            bonds = infer_bonds(elems, xyz_arr)

            stem = f"rank{rank:02d}_idx{idx:05d}_vmin{score:.2f}"
            complex_pdb = args.out_dir / f"{stem}_complex.pdb"
            write_full_protein_pdb(
                complex_pdb, args.receptor_pdb, elems, xyz_arr, ligand_bonds=bonds
            )
            (args.out_dir / f"{stem}_ligand.sdf").write_text(lig_sdf.read_text())

        formula = "".join(
            f"{e}{elems.count(e)}" for e in sorted(set(elems))
        )
        summary.append({
            "rank": rank, "idx": idx, "vina_min": round(score, 2),
            "vina_score_as_is": row["score_as_is"], "n_heavy": len(elems),
            "formula": formula, "file": complex_pdb.name,
        })
        logger.info(
            "  rank %2d  idx %5d  Vina Min %6.2f  %s", rank, idx, score, formula
        )

    with (args.out_dir / "top_poses_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    logger.info("Wrote %d complexes + summary to %s", len(summary), args.out_dir)


if __name__ == "__main__":
    main()

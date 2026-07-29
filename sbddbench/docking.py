"""Binding-affinity proxy via AutoDock Vina (category 2): Score / Min / Dock.

Following the standard SBDD protocol (TargetDiff, DiffSBDD, DiffGui all report
these three, separately):

* **Vina Score** (``--score_only``)  — the generated pose, scored as-is. Measures
  how good the *pose the model produced* is.
* **Vina Min**   (``--local_only``)  — local minimisation from the generated pose,
  then score. Measures whether a small relaxation rescues the pose.
* **Vina Dock**  (full search, ``--exhaustiveness``) — re-dock the molecule into
  the pocket from scratch; best mode. Measures whether the *molecule itself* can
  bind, independent of the generated pose.

Reporting all three (and their gaps) is the point: a model strong only on Vina
Dock produced bad poses that re-docking rescued; the Score↔Dock gap exposes it.

Ligand prep mirrors the in-house pipeline: Open Babel perceives bonds from the
generated 3D coordinates (largest fragment, ``-r``), protonates at pH 7.4, and
assigns Gasteiger charges. The generated heavy-atom coordinates are preserved,
so Score / Min act on the real generated geometry.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

from sbddbench import paths

_SCORE_RE = re.compile(
    r"Estimated Free Energy of Binding\s*:\s*(-?\d+\.?\d*)", re.IGNORECASE
)
_VINA_RESULT_RE = re.compile(r"REMARK VINA RESULT:\s*(-?\d+\.?\d*)")

_CFG: dict = {}


def _init_worker(cfg: dict) -> None:
    global _CFG  # noqa: PLW0603
    _CFG = cfg


def _write_xyz(path: Path, elements: list[str], coords) -> None:
    lines = [str(len(elements)), "generated"]
    lines += [
        f"{el} {x:.4f} {y:.4f} {z:.4f}"
        for el, (x, y, z) in zip(elements, coords, strict=True)
    ]
    path.write_text("\n".join(lines) + "\n")


def _read_pdbqt_heavy(path: Path) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    for ln in path.read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        atype = ln[77:79].strip() or ln.split()[-1]
        if atype.upper().startswith("H"):
            continue
        out.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    return out


def _heavy_rmsd(a, b) -> float | None:
    if not a or len(a) != len(b):
        return None
    sq = sum(
        (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2
        for (ax, ay, az), (bx, by, bz) in zip(a, b, strict=True)
    )
    return (sq / len(a)) ** 0.5


def _run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)


def _parse_score(stdout: str) -> float | None:
    m = _SCORE_RE.search(stdout)
    return float(m.group(1)) if m else None


def _best_dock_score(pdbqt: Path) -> float | None:
    if not pdbqt.exists():
        return None
    scores = [float(m.group(1)) for m in _VINA_RESULT_RE.finditer(pdbqt.read_text())]
    return min(scores) if scores else None


def _ligand_box(coords, box: dict | None, min_size: float, margin: float = 8.0):
    """Search box for Score/Min: contains the generated ligand. Falls back to the
    target pocket box centre if the ligand has zero extent."""
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]
    n = len(coords)
    cx, cy, cz = sum(xs) / n, sum(ys) / n, sum(zs) / n
    sx = max(min_size, (max(xs) - min(xs)) + margin)
    sy = max(min_size, (max(ys) - min(ys)) + margin)
    sz = max(min_size, (max(zs) - min(zs)) + margin)
    return (cx, cy, cz), (sx, sy, sz)


def dock_one(record: dict) -> dict:
    cfg = _CFG
    elements, coords = record["elements"], record["coords"]
    row: dict = {
        "idx": record["idx"], "tag": record.get("tag", ""),
        "n_atoms": len(elements), "n_atoms_docked": None, "dock_ok": False,
        "vina_score": None, "vina_min": None, "vina_dock": None,
        "min_rmsd": None, "reason": "",
    }
    if len(elements) < 3:
        row["reason"] = "too few atoms"
        return row

    (cx, cy, cz), (sx, sy, sz) = _ligand_box(coords, cfg.get("box"), cfg["box_size"])
    lig_box = [
        "--center_x", f"{cx:.3f}", "--center_y", f"{cy:.3f}", "--center_z", f"{cz:.3f}",
        "--size_x", f"{sx:.3f}", "--size_y", f"{sy:.3f}", "--size_z", f"{sz:.3f}",
    ]
    # Full-search box = the fixed pocket box (same for every ligand at a target),
    # so Vina Dock searches the real pocket, not a ligand-local cube.
    tbox = cfg.get("box")
    if tbox:
        c, s = tbox["center"], tbox["size"]
        dock_box = [
            "--center_x", f"{c[0]:.3f}", "--center_y", f"{c[1]:.3f}", "--center_z", f"{c[2]:.3f}",
            "--size_x", f"{s[0]:.3f}", "--size_y", f"{s[1]:.3f}", "--size_z", f"{s[2]:.3f}",
        ]
    else:
        dock_box = lig_box
    common = ["--receptor", cfg["receptor"], "--cpu", "1", "--seed", "1"]

    with tempfile.TemporaryDirectory(dir=cfg["tmp_dir"]) as td:
        tdp = Path(td)
        xyz, pdbqt = tdp / "lig.xyz", tdp / "lig.pdbqt"
        _write_xyz(xyz, elements, coords)
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
            if "score" in cfg["modes"]:
                r = _run([cfg["vina"], "--ligand", str(pdbqt), "--score_only", *common, *lig_box])
                row["vina_score"] = _parse_score(r.stdout)
            if "min" in cfg["modes"]:
                optp = tdp / "opt.pdbqt"
                r = _run([cfg["vina"], "--ligand", str(pdbqt), "--local_only",
                          "--out", str(optp), *common, *lig_box])
                row["vina_min"] = _parse_score(r.stdout)
                if optp.exists():
                    row["min_rmsd"] = _heavy_rmsd(
                        _read_pdbqt_heavy(pdbqt), _read_pdbqt_heavy(optp)
                    )
            if "dock" in cfg["modes"]:
                dockp = tdp / "dock.pdbqt"
                _run([cfg["vina"], "--ligand", str(pdbqt),
                      "--exhaustiveness", str(cfg["exhaustiveness"]),
                      "--out", str(dockp), *common, *dock_box])
                row["vina_dock"] = _best_dock_score(dockp)
        except subprocess.TimeoutExpired:
            row["reason"] = "vina timeout"
            return row
    row["dock_ok"] = any(
        row[k] is not None for k in ("vina_score", "vina_min", "vina_dock")
    )
    if not row["dock_ok"]:
        row["reason"] = row["reason"] or "vina produced no score"
    return row


def dock_generated(
    gen_mols,
    receptor_pdbqt: str | Path,
    box: dict | None,
    *,
    modes: tuple[str, ...] = ("score", "min", "dock"),
    workers: int | None = None,
    box_size: float = 22.5,
    exhaustiveness: int = 8,
    vina: str | None = None,
    obabel: str | None = None,
    tmp_dir: str | None = None,
) -> list[dict]:
    """Dock a list of :class:`sbddbench.molio.GenMol`. Returns one row per molecule.

    Only molecules with real elements and ≥3 atoms are docked; others get a row
    with ``dock_ok=False`` and a reason.
    """
    from sbddbench.molio import REAL_ELEMENTS

    workers = workers or (os.cpu_count() or 8)
    tmp_dir = tmp_dir or os.environ.get("T4TMPDIR") or str(Path.home() / "tmpdir")
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    cfg = {
        "receptor": str(receptor_pdbqt),
        "vina": vina or paths.VINA,
        "obabel": obabel or paths.OBABEL,
        "box": box,
        "box_size": box_size,
        "exhaustiveness": exhaustiveness,
        "modes": modes,
        "tmp_dir": tmp_dir,
    }
    records = [
        {"idx": g.idx, "tag": g.tag, "elements": g.elements,
         "coords": [list(map(float, c)) for c in g.coords]}
        for g in gen_mols
        if g.elements and not g.has_unknown_element and len(g.elements) >= 3
        and all(e in REAL_ELEMENTS for e in g.elements)
    ]
    if not records:
        return []
    with Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        return list(pool.imap_unordered(dock_one, records, chunksize=4))

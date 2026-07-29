"""PoseBusters chemical-validity checks on reconstructed ligands.

RMSD says how far the atoms moved; it does not say whether what came back is
still a molecule. A tokenizer can land every atom within a few tenths of an
Angstrom and still stretch a bond, flatten a ring the wrong way, or drop two
atoms onto the same point. PoseBusters' ``mol`` checks catch exactly that.

Because this is a *reconstruction*, not a generation, we already know the
molecule's bond graph: we rebuild the RDKit mol from the reference bonds plus
the reconstructed coordinates. No bond perception is involved, so a failure is
attributable to the tokenizer's geometry and nothing else.

Kept out of :mod:`plbench.metrics` on purpose -- that module is pure NumPy and
fast, while this one is slow (the energy-ratio check regenerates conformers,
~1 s/molecule) and pulls in RDKit + PoseBusters.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import pandas as pd

# SDF bond type code -> RDKit bond order.
_BOND_ORDER = {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE", 4: "AROMATIC"}


@lru_cache(maxsize=1)
def _buster():
    from posebusters import PoseBusters

    return PoseBusters(config="mol")


def build_mol(
    elements: Sequence[str],
    bonds: Sequence[tuple[int, int]],
    bond_orders: Sequence[int],
    coords: np.ndarray,
):
    """RDKit mol from a known bond graph plus coordinates; None if unsanitizable."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    rw = Chem.RWMol()
    for symbol in elements:
        rw.AddAtom(Chem.Atom(str(symbol)))
    for (i, j), order in zip(bonds, bond_orders, strict=True):
        if rw.GetBondBetweenAtoms(int(i), int(j)) is None:
            rw.AddBond(
                int(i),
                int(j),
                getattr(Chem.BondType, _BOND_ORDER.get(int(order), "SINGLE")),
            )
    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, (x, y, z) in enumerate(np.asarray(coords, dtype=float)):
        conf.SetAtomPosition(idx, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    try:
        Chem.SanitizeMol(mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError):
        return None
    return mol


_MAX_WORKERS = 12

_CACHE: dict[str, dict[str, float]] = {}


def _key(elements, bonds, coords) -> str:
    """Content hash of a conformer, so identical molecules are checked once."""
    payload = (
        ",".join(map(str, elements)).encode()
        + b"|"
        + np.asarray(bonds, dtype=np.int64).tobytes()
        + b"|"
        + np.round(np.asarray(coords, dtype=float), 3).tobytes()
    )
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def check(
    elements: Sequence[str],
    bonds: Sequence[tuple[int, int]],
    bond_orders: Sequence[int],
    coords: np.ndarray,
) -> dict[str, float]:
    """Run the PoseBusters ``mol`` suite; ``pb_valid`` is 1.0 iff all pass.

    Memoized on the conformer's contents: every model in a run is scored against
    the same crystal reference, so without this the reference would be busted
    once per model for no new information -- and each check costs about a second.

    An unsanitizable molecule scores 0 rather than raising: that is a genuine
    tokenizer failure and belongs in the table, not in an exception.
    """
    key = _key(elements, bonds, coords)
    if key in _CACHE:
        return _CACHE[key]
    mol = build_mol(elements, bonds, bond_orders, coords)
    if mol is None:
        result = {"pb_valid": 0.0, "pb_sanitization": 0.0}
    else:
        checks = _buster().bust([mol], None, None).iloc[0]
        result = {
            # skipna: a check PoseBusters could not run (e.g. the energy ratio
            # when UFF has no parameters for the molecule) is not evidence of a
            # bad conformer, so it must not count against it.
            "pb_valid": float(bool(checks.all(skipna=True))),
            # ... and it must not count *for* it either. bool(nan) is True, so
            # writing the raw value would silently record an unevaluated check
            # as a pass; keep it NaN so it drops out of the mean instead.
            **{
                f"pb_{name}": (np.nan if pd.isna(value) else float(bool(value)))
                for name, value in checks.items()
            },
        }
    _CACHE[key] = result
    return result


def _worker(job):
    """Pool entry point: returns (cache key, checks) for one conformer."""
    elements, bonds, bond_orders, coords = job
    quiet_rdkit()
    key = _key(elements, bonds, coords)
    mol = build_mol(elements, bonds, bond_orders, coords)
    if mol is None:
        return key, {"pb_valid": 0.0, "pb_sanitization": 0.0}
    checks = _buster().bust([mol], None, None).iloc[0]
    return key, {
        "pb_valid": float(bool(checks.all(skipna=True))),
        **{
            f"pb_{name}": (np.nan if pd.isna(value) else float(bool(value)))
            for name, value in checks.items()
        },
    }


def prefetch(jobs: Sequence[tuple], workers: int | None = None) -> int:
    """Check many conformers in parallel, filling the cache for later lookups.

    PoseBusters costs ~25 s per drug-like ligand -- the energy-ratio check
    regenerates a conformer ensemble -- so a serial pass over nine tokenizer
    arms x 303 complexes runs to roughly seventeen hours and does not fit in a
    job. The work is embarrassingly parallel and CPU-bound, so it goes to a
    process pool; :func:`check` then finds everything already cached and the
    row-building code needs no changes.

    Returns the number of conformers actually computed (cache hits are skipped).
    """
    import multiprocessing as mp
    import os

    pending, seen = [], set()
    for job in jobs:
        elements, bonds, _orders, coords = job
        key = _key(elements, bonds, coords)
        if key in _CACHE or key in seen:
            continue
        seen.add(key)
        pending.append(job)
    if not pending:
        return 0
    # sched_getaffinity, NOT cpu_count: on a scheduler-allocated node cpu_count
    # reports the whole machine (384 here) while the job only owns a slice (8 on
    # gpu_1). Sizing the pool from cpu_count spawns ~48x more PoseBusters
    # processes than there are cores, and the thrashing makes it far slower than
    # running serially -- which is exactly how one job burned 5 h on a single arm.
    available = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 2)
    )
    # Capped, and overridable. The 2500-thread ceiling is per USER and shared
    # with every other job that account is running, so "cores I can see" is an
    # upper bound on what is safe rather than a target -- filling it on a shared
    # interactive node killed a run mid-PoseBusters. On a dedicated batch node,
    # raise it with PLBENCH_PB_WORKERS.
    env_workers = os.environ.get("PLBENCH_PB_WORKERS")
    default = int(env_workers) if env_workers else min(available - 1, _MAX_WORKERS)
    workers = workers or max(1, min(default, len(pending)))
    with mp.get_context("spawn").Pool(workers) as pool:
        for key, result in pool.imap_unordered(_worker, pending, chunksize=1):
            _CACHE[key] = result
    return len(pending)


def quiet_rdkit() -> None:
    """Silence RDKit's per-molecule warnings (a full run emits thousands)."""
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

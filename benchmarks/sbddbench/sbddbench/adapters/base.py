"""Common interface every generation adapter implements.

An adapter's single job: given a :class:`~sbddbench.types.Target` (a prepared
pocket), drive its model — in the model's *own* interpreter, as a subprocess —
to sample ``n_samples`` ligands, and normalise whatever the model writes into a
single ``generated.sdf`` (one 3D molecule per entry). The bench evaluator then
reads that SDF; no adapter imports a generative model into the bench env.
"""

from __future__ import annotations

import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path

from rdkit import Chem

from sbddbench.types import GenResult, Target


class GenerativeModel(ABC):
    """Sample ligands for a protein pocket."""

    name: str = "base"
    #: does the model need an explicit ≤10 Å pocket PDB (vs full receptor)?
    needs_pocket_pdb: bool = False

    def setup(self) -> None:  # noqa: B027 - optional hook
        """Validate weights / interpreter exist. Idempotent."""

    @abstractmethod
    def generate(self, target: Target, n_samples: int, out_dir: Path) -> GenResult:
        """Generate ``n_samples`` ligands for ``target``; write
        ``out_dir/generated.sdf``. Must not raise: on failure return a
        ``GenResult`` with ``ok=False`` and an ``error`` message."""

    # -- helpers ----------------------------------------------------------
    def _timed_generate(self, target: Target, n_samples: int, out_dir: Path) -> GenResult:
        t0 = time.perf_counter()
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = self.generate(target, n_samples, out_dir)
        except Exception as exc:  # noqa: BLE001 - adapters must never crash the run
            result = GenResult(model=self.name, target_id=target.target_id,
                               ok=False, error=repr(exc))
        if result.runtime_s is None:
            result.runtime_s = time.perf_counter() - t0
        return result

    @staticmethod
    def _run(cmd, cwd=None, env=None, timeout=None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(c) for c in cmd], cwd=cwd and str(cwd), env=env,
            capture_output=True, text=True, check=False, timeout=timeout,
        )

    @staticmethod
    def _env_for(python, **extra) -> dict:
        """Subprocess environment for a conda/venv interpreter.

        Conda's OpenBabel python bindings can't find their plugins without
        ``BABEL_LIBDIR`` / ``BABEL_DATADIR`` (they silently read 0 atoms
        otherwise), which breaks DiffSBDD's molecule builder. Point them at the
        interpreter's own openbabel install, then merge any ``extra`` vars.
        """
        import glob
        import os

        env = dict(os.environ)
        prefix = Path(python).resolve().parent.parent
        libdirs = sorted(glob.glob(str(prefix / "lib" / "openbabel" / "*")))
        datadirs = sorted(glob.glob(str(prefix / "share" / "openbabel" / "*")))
        if libdirs:
            env["BABEL_LIBDIR"] = libdirs[-1]
        if datadirs:
            env["BABEL_DATADIR"] = datadirs[-1]
        env.update({k: str(v) for k, v in extra.items()})
        return env

    @staticmethod
    def _collect_sdf_files(sdf_files, out_sdf: Path) -> int:
        """Concatenate a list of single-mol ``*.sdf`` files into one multi-mol
        ``generated.sdf``. Returns the number of molecules written."""
        n = 0
        writer = Chem.SDWriter(str(out_sdf))
        for p in sorted(sdf_files):
            for mol in Chem.SDMolSupplier(str(p), sanitize=False, removeHs=False):
                if mol is None:
                    continue
                mol.SetProp("_Name", Path(p).stem)
                writer.write(mol)
                n += 1
        writer.close()
        return n

    @classmethod
    def _collect_sdf_dir(cls, sdf_dir: Path, out_sdf: Path) -> int:
        """Concatenate every ``*.sdf`` (one mol each) under ``sdf_dir`` into one
        multi-mol ``generated.sdf``. Returns the number of molecules written."""
        files = [p for p in sorted(sdf_dir.glob("*.sdf")) if not p.name.startswith("traj_")]
        return cls._collect_sdf_files(files, out_sdf)

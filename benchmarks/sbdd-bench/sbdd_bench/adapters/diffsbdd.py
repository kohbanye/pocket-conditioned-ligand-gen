"""DiffSBDD adapter (Schneuing et al., Nat. Comput. Sci. 2024).

Drives the repo's ``generate_ligands.py`` in the DiffSBDD conda/uv env. The
pocket is defined from the reference ligand SDF (``--ref_ligand``, which DiffSBDD
accepts as an .sdf), the full receptor is passed as ``--pdbfile``, and ligands
are written straight to ``generated.sdf``. ``--sanitize`` keeps only RDKit-valid
molecules; ``--all_frags`` is off so the largest fragment is kept per sample.
"""

from __future__ import annotations

from pathlib import Path

from sbdd_bench import paths
from sbdd_bench.adapters.base import GenerativeModel
from sbdd_bench.types import GenResult, Target


class DiffSBDDAdapter(GenerativeModel):
    name = "diffsbdd"
    needs_pocket_pdb = False

    def __init__(self, ckpt=None, python=None, batch_size: int | None = None,
                 sanitize: bool = True, relax: bool = False, timesteps=None, **_):
        self.ckpt = Path(ckpt) if ckpt else paths.DIFFSBDD_CKPT
        self.python = python or paths.DIFFSBDD_PYTHON
        self.repo = paths.DIFFSBDD_REPO
        self.batch_size = batch_size
        self.sanitize = sanitize
        self.relax = relax
        self.timesteps = timesteps

    def setup(self) -> None:
        if not self.ckpt.exists():
            raise FileNotFoundError(
                f"DiffSBDD checkpoint missing: {self.ckpt}. Run scripts/fetch_weights.py --diffsbdd."
            )

    def generate(self, target: Target, n_samples: int, out_dir: Path) -> GenResult:
        self.setup()
        sdf = out_dir / "generated.sdf"
        # DiffSBDD requires n_samples % batch_size == 0.
        bs = self.batch_size or min(n_samples, 64)
        while n_samples % bs != 0:
            bs -= 1
        cmd = [
            self.python, self.repo / "generate_ligands.py", self.ckpt.resolve(),
            "--pdbfile", Path(target.receptor_pdb).resolve(),
            "--ref_ligand", Path(target.ref_ligand_sdf).resolve(),
            "--outfile", sdf.resolve(),
            "--n_samples", n_samples,
            "--batch_size", bs,
        ]
        if self.sanitize:
            cmd.append("--sanitize")
        if self.relax:
            cmd.append("--relax")
        if self.timesteps:
            cmd += ["--timesteps", self.timesteps]
        proc = self._run(cmd, cwd=self.repo, env=self._env_for(self.python))
        if not sdf.exists():
            return GenResult(self.name, target.target_id, ok=False,
                             n_requested=n_samples,
                             error=(proc.stderr or proc.stdout or "")[-2000:])
        from rdkit import Chem

        n = sum(1 for m in Chem.SDMolSupplier(str(sdf), sanitize=False) if m is not None)
        return GenResult(self.name, target.target_id, sdf=sdf,
                         n_requested=n_samples, n_generated=n)

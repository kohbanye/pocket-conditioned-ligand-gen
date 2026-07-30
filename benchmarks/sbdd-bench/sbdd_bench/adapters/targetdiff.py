"""TargetDiff adapter (Guan et al., ICLR 2023).

Drives the repo's ``scripts/sample_for_pocket.py`` in the TargetDiff conda env.
That script reads a ≤10 Å pocket PDB, samples ligands with the diffusion model,
reconstructs RDKit molecules, and writes one SDF per molecule into
``<result_path>/sdf/``. We generate a temp sampling config that points at the
pretrained checkpoint and the requested sample count, then concatenate the
per-molecule SDFs into a single ``generated.sdf``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from sbdd_bench import paths
from sbdd_bench.adapters.base import GenerativeModel
from sbdd_bench.types import GenResult, Target


class TargetDiffAdapter(GenerativeModel):
    name = "targetdiff"
    needs_pocket_pdb = True

    def __init__(self, ckpt=None, python=None, batch_size: int = 20,
                 num_steps: int = 1000, seed: int = 2021, **_):
        self.ckpt = Path(ckpt) if ckpt else paths.TARGETDIFF_CKPT
        self.python = python or paths.TARGETDIFF_PYTHON
        self.repo = paths.TARGETDIFF_REPO
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.seed = seed

    def setup(self) -> None:
        if not self.ckpt.exists():
            raise FileNotFoundError(
                f"TargetDiff checkpoint missing: {self.ckpt}. Run scripts/fetch_weights.py --targetdiff."
            )

    def generate(self, target: Target, n_samples: int, out_dir: Path) -> GenResult:
        self.setup()
        if target.pocket_pdb is None:
            return GenResult(self.name, target.target_id, ok=False,
                             error="targetdiff needs target.pocket_pdb (≤10 Å pocket)")
        cfg = {
            "model": {"checkpoint": str(self.ckpt.resolve())},
            "sample": {
                "seed": self.seed, "num_samples": n_samples,
                "num_steps": self.num_steps, "pos_only": False,
                "center_pos_mode": "protein", "sample_num_atoms": "prior",
            },
        }
        cfg_path = out_dir / "sampling.yml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        result_path = out_dir / "td_out"
        cmd = [
            self.python, self.repo / "scripts" / "sample_for_pocket.py",
            cfg_path.resolve(),
            "--pdb_path", Path(target.pocket_pdb).resolve(),
            "--num_samples", n_samples,
            "--batch_size", self.batch_size,
            "--result_path", result_path.resolve(),
        ]
        proc = self._run(cmd, cwd=self.repo,
                         env=self._env_for(self.python, PYTHONPATH=str(self.repo)))
        sdf_dir = result_path / "sdf"
        sdf = out_dir / "generated.sdf"
        if not sdf_dir.exists():
            return GenResult(self.name, target.target_id, ok=False,
                             n_requested=n_samples,
                             error=(proc.stderr or proc.stdout or "")[-2000:])
        n = self._collect_sdf_dir(sdf_dir, sdf)
        return GenResult(self.name, target.target_id, sdf=sdf,
                         n_requested=n_samples, n_generated=n)

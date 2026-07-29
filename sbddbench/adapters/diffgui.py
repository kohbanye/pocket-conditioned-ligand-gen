"""DiffGui adapter (Hu et al., J. Cheminform. 2024).

Drives the repo's ``scripts/sample.py`` in the DiffGui conda env. DiffGui is
config-driven: we write a temp ``diffgui.yml`` for de-novo pocket sampling that
points at the main checkpoint and the bond-predictor (used as guidance), sets
``gen_mode=denovo`` / ``mode=pocket`` and the pocket PDB, then collect the
per-molecule SDFs it writes under ``<outdir>/diffgui_*/*_SDF/``.

Note: DiffGui scores each molecule with QuickVina during sampling, so the
sampling env must provide the qvina binary (see scripts/setup_envs.sh).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from sbddbench import paths
from sbddbench.adapters.base import GenerativeModel
from sbddbench.types import GenResult, Target


class DiffGuiAdapter(GenerativeModel):
    name = "diffgui"
    needs_pocket_pdb = True

    def __init__(self, ckpt=None, bond_ckpt=None, python=None, batch_size: int = 4,
                 seed: int = 2023, gui_strength: float = 3.0, **_):
        self.ckpt = Path(ckpt) if ckpt else paths.DIFFGUI_CKPT
        self.bond_ckpt = Path(bond_ckpt) if bond_ckpt else (self.ckpt.parent / "bond_trained.pt")
        self.python = python or paths.DIFFGUI_PYTHON
        self.repo = paths.DIFFGUI_REPO
        self.batch_size = batch_size
        self.seed = seed
        self.gui_strength = gui_strength

    def setup(self) -> None:
        for p, what in [(self.ckpt, "checkpoint"), (self.bond_ckpt, "bond-predictor checkpoint")]:
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"DiffGui {what} missing: {p}. Run scripts/fetch_weights.py --diffgui."
                )

    def generate(self, target: Target, n_samples: int, out_dir: Path) -> GenResult:
        self.setup()
        if target.pocket_pdb is None:
            return GenResult(self.name, target.target_id, ok=False,
                             error="diffgui needs target.pocket_pdb (≤10 Å pocket)")
        cfg = {
            "model": {
                "checkpoint": str(self.ckpt.resolve()),
                "target": str(Path(target.pocket_pdb).resolve()),
                "ligand": "None", "frag": "None", "gen_mode": "denovo",
                "logp": 2.0, "tpsa": 100, "sa": 1.0, "qed": 0.8, "aff": 12.0,
            },
            "bond_predictor": str(self.bond_ckpt.resolve()),
            "sample": {
                "seed": self.seed, "batch_size": self.batch_size, "num_mols": n_samples,
                "save_traj_prob": 0.0, "sample": True, "sample_method": "priori",
                "mode": "pocket", "test_id": 0, "add_edge": None,
                "gui_strength": self.gui_strength,
                "guidance": ["uncertainty", 1.0e-4],
            },
            "data": {
                "name": "protein_ligand", "dataset": "crossdocked",
                "transform": {"ligand_atom_mode": "aromatic", "random_rot": False, "sample": False},
            },
        }
        cfg_path = out_dir / "diffgui.yml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        cmd = [
            self.python, self.repo / "scripts" / "sample.py",
            "--config", cfg_path.resolve(),
            "--outdir", out_dir.resolve(),
        ]
        proc = self._run(cmd, cwd=self.repo,
                         env=self._env_for(self.python, PYTHONPATH=str(self.repo)))
        sdf = out_dir / "generated.sdf"
        sdf_files = [p for p in out_dir.rglob("*_SDF/*.sdf") if not p.name.startswith("traj_")]
        if not sdf_files:
            return GenResult(self.name, target.target_id, ok=False,
                             n_requested=n_samples,
                             error=(proc.stderr or proc.stdout or "")[-2000:])
        n = self._collect_sdf_files(sdf_files, sdf)
        return GenResult(self.name, target.target_id, sdf=sdf,
                         n_requested=n_samples, n_generated=n)

"""FLOWR adapter (Cremer et al., arXiv 2504.10564).

Drives ``flowr.gen.generate_from_pdb`` in FLOWR's own conda environment. It
takes the receptor PDB and the reference ligand SDF -- the same two files every
other adapter here is handed -- and cuts its own pocket around the ligand, so
nothing about the target set has to be prepared differently for it.

**Why run it rather than cite it.** The paper's published numbers are PB 0.92
and Vina Score -6.29; it does not give Vina Min or Vina Dock, which are two of
the three numbers this benchmark reports. Running it puts all three on the same
100 pockets, through the same Vina settings and the same receptor pdbqt files,
so the comparison stops depending on what a paper happened to tabulate.

**What the comparison does and does not hold fixed.** Fixed: the test pockets,
the reference ligands, the scorer. *Not* fixed: the training data -- FLOWR is
trained on SPINDR and ProLIT on CrossDocked2020. That is a real difference and
belongs in the table's caption, not hidden by the fact that both are evaluated
here.

FLOWR writes ``samples_<target>.sdf`` into its ``--save_dir``; this renames that
to the ``generated.sdf`` every other adapter produces.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from sbdd_bench import paths
from sbdd_bench.adapters.base import GenerativeModel
from sbdd_bench.types import GenResult, Target


class FlowrAdapter(GenerativeModel):
    name = "flowr"
    needs_pocket_pdb = False

    def __init__(  # noqa: PLR0913
        self,
        ckpt: str | Path | None = None,
        python: str | Path | None = None,
        pocket_cutoff: float = 6.0,
        batch_cost: int | None = None,
        seed: int = 42,
        **_: object,
    ) -> None:
        self.ckpt = Path(ckpt) if ckpt else paths.FLOWR_CKPT
        self.python = str(python or paths.FLOWR_PYTHON)
        self.repo = paths.FLOWR_REPO
        self.pocket_cutoff = pocket_cutoff
        # Six of the 100 pockets ran the GPU out of memory at the default 100
        # -- all of them large. The knob is a batch SIZE in FLOWR's own units,
        # so lowering it costs wall-clock and changes nothing about what the
        # model produces.
        self.batch_cost = int(
            batch_cost if batch_cost is not None
            else os.environ.get("SBDD_FLOWR_BATCH_COST", 100)
        )
        self.seed = seed

    def setup(self) -> None:
        if not self.ckpt.exists():
            msg = (
                f"FLOWR checkpoint missing: {self.ckpt}. Download flowr_noHs.ckpt "
                "from https://zenodo.org/records/15737419 into "
                f"{self.ckpt.parent}."
            )
            raise FileNotFoundError(msg)
        if not Path(self.python).exists():
            msg = (
                f"FLOWR python missing: {self.python}. Create the environment "
                f"from {self.repo / 'environment.yml'} and point "
                "SBDD_FLOWR_PYTHON at its interpreter."
            )
            raise FileNotFoundError(msg)

    def generate(self, target: Target, n_samples: int, out_dir: Path) -> GenResult:
        self.setup()
        out_dir.mkdir(parents=True, exist_ok=True)
        # The repo is imported as a package from its own root, and its env is a
        # separate interpreter, so PYTHONPATH is where the two meet.
        env = dict(os.environ, PYTHONPATH=str(self.repo))
        # ``generate_from_pdb`` pulls in ``pymol2``, and the PyMOL in FLOWR's
        # environment is Schrodinger's build, which stats
        # ``$SCHRODINGER/licenses`` while loading. Where /opt/schrodinger exists
        # but is root-only, that throws a C++ filesystem_error and aborts the
        # interpreter outright -- no Python traceback, just a core dump, which
        # is why this took a while to find. It is not enough to point
        # SCHRODINGER somewhere readable: the ``licenses`` directory has to
        # exist. Nothing here uses a licensed feature; the directory is empty.
        licenses = paths.WEIGHTS_DIR / "flowr" / "schrodinger" / "licenses"
        licenses.mkdir(parents=True, exist_ok=True)
        env["SCHRODINGER"] = str(licenses.parent)
        cmd = [
            self.python, "-m", "flowr.gen.generate_from_pdb",
            "--pdb_file", str(target.receptor_pdb),
            "--ligand_file", str(target.ref_ligand_sdf),
            "--ckpt_path", str(self.ckpt.resolve()),
            "--save_dir", str(out_dir.resolve()),
            "--arch", "pocket",
            "--pocket_type", "holo",
            "--cut_pocket",
            "--pocket_cutoff", str(self.pocket_cutoff),
            "--gpus", "1",
            "--batch_cost", str(self.batch_cost),
            "--max_sample_iter", "20",
            "--sample_n_molecules_per_target", str(n_samples),
            "--categorical_strategy", "uniform-sample",
            "--sample_mol_sizes",
            "--seed", str(self.seed),
        ]
        proc = self._run(cmd, cwd=self.repo, env=env)

        # FLOWR names its output after the target; the benchmark expects one
        # fixed name per directory.
        produced = sorted(out_dir.glob("samples_*.sdf"))
        produced = [p for p in produced if "protonated" not in p.name]
        sdf = out_dir / "generated.sdf"
        if produced:
            shutil.move(str(produced[0]), sdf)
        if not sdf.exists():
            return GenResult(
                self.name, target.target_id, ok=False, n_requested=n_samples,
                error=(proc.stderr or proc.stdout or "")[-2000:],
            )
        from rdkit import Chem  # noqa: PLC0415

        n = sum(
            1 for m in Chem.SDMolSupplier(str(sdf), sanitize=False) if m is not None
        )
        return GenResult(
            self.name, target.target_id, sdf=sdf, n_requested=n_samples, n_generated=n
        )

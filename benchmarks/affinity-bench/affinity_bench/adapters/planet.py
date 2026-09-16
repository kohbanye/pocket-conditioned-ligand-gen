"""PLANET (Zhang et al., JCIM 2023) via its released checkpoint.

NOT interaction-featurised, but also NOT pose-aware, which is why this row sits
beside Boltz-2 rather than beside the scoring functions. PLANET encodes the
pocket as a graph over residues and alpha-carbon coordinates and the ligand as a
2D graph; the authors state the model "does not take three-dimensional
protein-ligand structures as input". The protein-ligand contact map it predicts
is a multi-task TRAINING target scored against ground truth, never an input
feature, so it passes the interaction-free test while failing the same-input
test: the crystal ligand fixes the pocket centre and nothing more.

Leak handling is explicit upstream: PDBbind general v2020 with "the 285
protein-ligand complexes included in the CASF-2016 test set ... removed from our
dataset to avoid overlap".

Ligands are read from MOL2, not SDF. 92 of the 285 CASF ligands fail strict
RDKit sanitisation from their SDF; all 285 parse strictly from MOL2. That
matters more here than for any other baseline because PLANET consumes a chemical
GRAPH -- measured, a partial-sanitisation fallback cost it 0.0855 R on those 92
complexes while GIGN, OnionNet-2, Sfcnn and Vina, which read coordinates, moved
by at most 0.014. See scripts/planet_score_cli.py.

Upstream's PLANET_run.py is not used: its scoring loop wraps each batch in
``except: continue``, which would have silently dropped those 92 complexes and
reported a 193-complex number as CASF-2016.

Reproduced here at R 0.7870 against a published 0.811 (preprint) / 0.824
(journal). The residual is unexplained; scoring the authors' own Maestro-prepared
bundle (core.tar.bz2) gives 0.7753, i.e. LOWER than our raw-CASF number, so
input preparation does not account for it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from affinity_bench import paths
from affinity_bench.adapters.base import DUMP_COLUMNS, AffinityBaseline

if TYPE_CHECKING:
    from collections.abc import Sequence

    from affinity_bench.inference import Complex

#: Shipped in the repository as an ordinary 18.5 MB file.
WEIGHTS_NAME = "PLANET.param"


class PlanetAdapter(AffinityBaseline):
    name = "planet"
    uses_explicit_contacts = False

    def __init__(
        self,
        python: str | None = None,
        repo: Path | None = None,
        ligand_source: str = "mol2",
    ) -> None:
        self.python = python or paths.PLANET_PYTHON
        self.repo = repo or paths.PLANET_REPO
        self.ligand_source = ligand_source

    def setup(self) -> None:
        if not self.repo.exists():
            msg = (
                f"PLANET checkout not found at {self.repo}. Clone "
                "https://github.com/ComputArtCMCG/PLANET there, or set "
                "AFFINITY_PLANET_REPO."
            )
            raise FileNotFoundError(msg)
        weights = self.repo / WEIGHTS_NAME
        if not weights.exists():
            msg = f"PLANET weights missing: {weights}"
            raise FileNotFoundError(msg)
        if not Path(self.python).exists():
            msg = (
                f"No PLANET interpreter at {self.python}. Build it with "
                "`cd third_party/PLANET && uv venv .venv-planet --python 3.11` "
                "then install torch (CPU is enough for 285 complexes), numpy, "
                "scipy, pandas, scikit-learn, rdkit and tqdm. Or set "
                "AFFINITY_PLANET_PYTHON."
            )
            raise RuntimeError(msg)

    def score(self, complexes: Sequence[Complex]) -> pd.DataFrame:
        """Score every complex, failing loudly if any is absent from the dump."""
        self.setup()
        work = paths.WORK_DIR / "planet"
        work.mkdir(parents=True, exist_ok=True)
        out = work / "scoring.csv"
        cli = paths.REPO_ROOT / "scripts" / "planet_score_cli.py"
        cmd = [
            str(self.python), str(cli),
            "--casf", str(paths.CASF_DIR),
            "--repo", str(self.repo),
            "--out", str(out),
            "--ligand-source", self.ligand_source,
        ]
        # cwd is the repo root: PLANET.load_parameters() resolves PLANET.param
        # by a bare relative path.
        subprocess.run(cmd, check=True, cwd=self.repo, stdout=sys.stderr)  # noqa: S603
        frame = pd.read_csv(out)
        self.check_coverage(frame, complexes)
        return frame[list(DUMP_COLUMNS)]

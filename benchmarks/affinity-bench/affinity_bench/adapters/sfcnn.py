"""Sfcnn (Wang, Wei and Xi, BMC Bioinformatics 2022) via its released weights.

NOT interaction-featurised, and this is why the row matters: Sfcnn crops one
joint 20x20x20 A grid centred on the ligand, holding protein and ligand atoms
together, and encodes each occupied voxel as a 28-channel one-hot atom type (14
protein, 14 ligand). The paper states it "simplified the featurization and only
used one-hot encoding of atomic types" -- no contact pairs, no distance bins, no
interaction fingerprint. It also consumes the full complex geometry rather than
a 2D ligand graph or a sequence.

That combination -- pose-aware AND handed no contacts -- makes it the first
like-for-like peer in this table for a tokenizer that is given no contacts
either. Every other pose-aware baseline here (GenScore, GIGN, OnionNet-2, Vina)
featurises interactions explicitly, and the one other contact-free row, Boltz-2,
never sees the crystal pose.

Leak handling is documented upstream: the refined set of PDBbind v2019 (4852)
minus "all of the overlaps between the test set and the refined set ... (266
overlaps)", leaving 4586 for training and validation.

Sfcnn pins TensorFlow 2 and OpenBabel, which cannot share this bench's
environment, so scoring runs in ``paths.SFCNN_PYTHON`` through
``scripts/sfcnn_score_cli.py``.

The published figure is 0.7928 on 283 complexes. This adapter scores all 285, on
our poses and labels, so the reproduced number differs -- that is the bench
contract, not a defect.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pandas as pd

from affinity_bench import paths
from affinity_bench.adapters.base import DUMP_COLUMNS, AffinityBaseline

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from affinity_bench.inference import Complex

#: Shipped in the repository, not tracked with git-lfs, so a plain checkout has
#: the real file rather than a pointer.
WEIGHTS_NAME = "weights_22_112-0.0083.h5"


class SfcnnAdapter(AffinityBaseline):
    name = "sfcnn"
    uses_explicit_contacts = False

    def __init__(
        self,
        python: str | None = None,
        repo: Path | None = None,
        ligand_format: str = "mol2",
    ) -> None:
        self.python = python or paths.SFCNN_PYTHON
        self.repo = repo or paths.SFCNN_REPO
        # Upstream's own default. CASF ships both .mol2 and .sdf; the format is
        # recorded because it is an input choice, not a detail.
        self.ligand_format = ligand_format

    def setup(self) -> None:
        if not self.repo.exists():
            msg = (
                f"Sfcnn checkout not found at {self.repo}. "
                "Clone https://github.com/bioinfocqupt/Sfcnn there, or set "
                "AFFINITY_SFCNN_REPO."
            )
            raise FileNotFoundError(msg)
        weights = self.repo / WEIGHTS_NAME
        if not weights.exists():
            msg = f"Sfcnn weights missing: {weights}"
            raise FileNotFoundError(msg)
        if self.python == "python":
            msg = (
                "No Sfcnn interpreter found. Build it with "
                "`sh benchmarks/affinity-bench/scripts/setup_sfcnn_env.sh`, "
                "or set AFFINITY_SFCNN_PYTHON."
            )
            raise RuntimeError(msg)

    def score(self, complexes: Sequence[Complex]) -> pd.DataFrame:
        """Score every complex, failing loudly if any is absent from the dump."""
        self.setup()
        work = paths.WORK_DIR / "sfcnn"
        work.mkdir(parents=True, exist_ok=True)
        out = work / "scoring.csv"
        cli = paths.REPO_ROOT / "scripts" / "sfcnn_score_cli.py"
        cmd = [
            self.python, str(cli),
            "--casf", str(paths.CASF_DIR),
            "--repo", str(self.repo),
            "--out", str(out),
            "--ligand-format", self.ligand_format,
        ]
        # cwd is the repo root: build_model() loads the .h5 by a bare relative
        # path, so anywhere else fails inside Keras with an unhelpful error.
        subprocess.run(cmd, check=True, cwd=self.repo, stdout=sys.stderr)  # noqa: S603
        frame = pd.read_csv(out)
        self.check_coverage(frame, complexes)
        return frame[list(DUMP_COLUMNS)]

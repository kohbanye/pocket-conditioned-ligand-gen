"""GenScore (Shen et al., Chem Sci 2023) via its released GatedGCN checkpoint.

Interaction-featurised: the score comes from learned protein-ligand distance
distributions.

GenScore pins a DGL build that cannot share this bench's environment, so scoring
runs in ``paths.GENSCORE_PYTHON`` through ``scripts/genscore_score_cli.py``.

This is the one baseline whose harness predates the submodule layout: the
working CASF driver lived in a sibling checkout alongside a conda environment
that is expensive to rebuild. The CLI now imports GenScore from the submodule,
while ``paths.GENSCORE_PYTHON`` still falls back to that environment's
interpreter -- the source is pinned here, the interpreter is wherever DGL works.
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

    from affinity_bench.inference import Complex

#: The checkpoint every GenScore number in this bench was measured with.
DEFAULT_CHECKPOINT = "GatedGCN_ft_1.0_1.pth"


class GenScoreAdapter(AffinityBaseline):
    name = "genscore"
    uses_explicit_contacts = True

    def __init__(
        self,
        python: str | None = None,
        checkpoint: str = DEFAULT_CHECKPOINT,
        encoder: str = "gatedgcn",
    ) -> None:
        self.python = python or paths.GENSCORE_PYTHON
        self.checkpoint = checkpoint
        self.encoder = encoder

    def setup(self) -> None:
        if not paths.GENSCORE_REPO.exists():
            msg = (
                f"GenScore submodule missing at {paths.GENSCORE_REPO}. "
                "Run: git submodule update --init third_party/GenScore"
            )
            raise FileNotFoundError(msg)
        ckpt = paths.GENSCORE_REPO / "trained_models" / self.checkpoint
        if not ckpt.exists():
            msg = f"GenScore checkpoint missing: {ckpt}"
            raise FileNotFoundError(msg)

    def score(self, complexes: Sequence[Complex]) -> pd.DataFrame:
        self.setup()
        work = paths.WORK_DIR / "genscore"
        out = work / "predictions" / "genscore.csv"
        cli = paths.REPO_ROOT / "scripts" / "genscore_score_cli.py"
        cmd = [
            self.python, str(cli),
            "--casf", str(paths.CASF_DIR),
            "--repo", str(paths.GENSCORE_REPO),
            "--work", str(work),
            "--checkpoint", self.checkpoint,
            "--encoder", self.encoder,
            "--out", str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
            msg = f"genscore_score_cli failed (rc={proc.returncode})"
            raise RuntimeError(msg)
        frame = pd.read_csv(out)
        # 4k18 and 4mme fail graph construction upstream and were recovered by
        # repairing their inputs; a run that silently drops them is a 283-complex
        # comparison wearing a 285-complex label.
        self.check_coverage(frame, complexes)
        return frame[list(DUMP_COLUMNS)]

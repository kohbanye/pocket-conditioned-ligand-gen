"""RTMScore (Shen et al., J Med Chem 2022) via its released checkpoints.

Interaction-featurised: the score is the likelihood of the observed
protein-ligand atom-pair distances under a learned mixture density, over a
residue-level pocket graph.

RTMScore pins a DGL build that cannot share this bench's environment, so scoring
runs in ``paths.RTMSCORE_PYTHON`` through ``scripts/casf_pose_score_cli.py``.

Three checkpoints ship with the repository. ``model1`` is the one the paper
reports and the default here; the others are available by name, but choosing
among them by their CASF result would be selection on the test set, which is the
rule our own arms are held to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pose_rescoring_bench import paths
from pose_rescoring_bench.adapters import _dgl_driver
from pose_rescoring_bench.adapters.base import DUMP_COLUMNS, PoseBaseline

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


class RTMScoreAdapter(PoseBaseline):
    name = "rtmscore"
    uses_explicit_contacts = True

    def __init__(
        self,
        python: str | None = None,
        checkpoint: str = paths.RTMSCORE_CHECKPOINT,
        num_workers: int = 4,
        allow_missing: Sequence[str] = (),
    ) -> None:
        self.python = python or paths.RTMSCORE_PYTHON
        self.checkpoint = checkpoint
        self.num_workers = num_workers
        self.allow_missing = allow_missing

    def setup(self) -> None:
        _dgl_driver.require_repo(paths.RTMSCORE_REPO, "RTMScore", self.checkpoint)

    def score(self, targets: Sequence[str]) -> pd.DataFrame:
        self.setup()
        stem = self.checkpoint.removesuffix(".pth")
        out = paths.WORK_DIR / "rtmscore" / "predictions" / f"{stem}.csv"
        frame = _dgl_driver.run_pose_cli(
            _dgl_driver.Backend(
                backend="rtmscore",
                python=self.python,
                repo=paths.RTMSCORE_REPO,
                checkpoint=self.checkpoint,
                num_workers=self.num_workers,
            ),
            targets,
            out,
        )
        self.check_coverage(frame, targets, self.allow_missing)
        return frame[list(DUMP_COLUMNS)]

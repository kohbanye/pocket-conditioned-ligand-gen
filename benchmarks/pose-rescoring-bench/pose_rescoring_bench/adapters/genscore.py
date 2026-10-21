"""GenScore (Shen et al., Chem Sci 2023) via its released GatedGCN checkpoint.

Interaction-featurised: the same distance-likelihood formulation as RTMScore, on
an atom-level pocket graph and retrained on a mixed docking/screening objective.

GenScore pins a PyTorch-Geometric build that cannot share this bench's
environment, so scoring runs in ``paths.GENSCORE_PYTHON`` through
``scripts/casf_pose_score_cli.py``.

``encoder`` selects the graph encoder and must match the checkpoint: the
released weights are shaped by it, and a mismatch loads them into the wrong
architecture. ``GatedGCN_ft_1.0_1.pth`` is a ``gatedgcn`` checkpoint, and it is
the one the affinity bench also measures, so the two tables describe the same
model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pose_rescoring_bench import paths
from pose_rescoring_bench.adapters import _dgl_driver
from pose_rescoring_bench.adapters.base import DUMP_COLUMNS, PoseBaseline

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


class GenScoreAdapter(PoseBaseline):
    name = "genscore"
    uses_explicit_contacts = True

    def __init__(
        self,
        python: str | None = None,
        checkpoint: str = paths.GENSCORE_CHECKPOINT,
        encoder: str = "gatedgcn",
        num_workers: int = 4,
        allow_missing: Sequence[str] = (),
    ) -> None:
        self.python = python or paths.GENSCORE_PYTHON
        self.checkpoint = checkpoint
        self.encoder = encoder
        self.num_workers = num_workers
        self.allow_missing = allow_missing

    def setup(self) -> None:
        _dgl_driver.require_repo(paths.GENSCORE_REPO, "GenScore", self.checkpoint)

    def score(self, targets: Sequence[str]) -> pd.DataFrame:
        self.setup()
        stem = self.checkpoint.removesuffix(".pth")
        out = paths.WORK_DIR / "genscore" / "predictions" / f"{stem}.csv"
        frame = _dgl_driver.run_pose_cli(
            _dgl_driver.Backend(
                backend="genscore",
                python=self.python,
                repo=paths.GENSCORE_REPO,
                checkpoint=self.checkpoint,
                encoder=self.encoder,
                num_workers=self.num_workers,
            ),
            targets,
            out,
        )
        self.check_coverage(frame, targets, self.allow_missing)
        return frame[list(DUMP_COLUMNS)]

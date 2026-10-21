"""AutoDock Vina as a pose-rescoring baseline, through ``--score_only``.

Not interaction-featurised in the learned sense and not trained at all: Vina is
an empirical function whose terms were fitted once, decades of complexes ago. It
is the reference row -- the number a rescorer has to beat to be worth running --
rather than a peer of the learned scorers.

Unlike the two DGL baselines this one needs no separate interpreter: the work is
done by the ``vina`` and ``obabel`` executables, which
``prolit.external_tools`` resolves, so the existing
``scripts/baseline_vina.py`` runs under this bench's own Python. The adapter
wraps that script rather than reimplementing it, because the script is what the
published Vina row was measured with.

It is slow -- roughly 28k poses over 285 targets -- so it wants a many-core CPU
node and ``workers`` set accordingly. Everything else here is instant by
comparison.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pandas as pd

from pose_rescoring_bench import paths
from pose_rescoring_bench.adapters.base import DUMP_COLUMNS, PoseBaseline

if TYPE_CHECKING:
    from collections.abc import Sequence


class VinaAdapter(PoseBaseline):
    name = "vina"
    uses_explicit_contacts = True
    is_empirical = True

    def __init__(
        self,
        python: str | None = None,
        workers: int = 40,
        allow_missing: Sequence[str] = (),
        *,
        exclude_native: bool = True,
    ) -> None:
        # Vina's driver imports prolit, so it runs under this bench's own
        # interpreter; there is no ``.venv-vina`` to fall back to.
        self.python = python or sys.executable
        self.workers = workers
        self.allow_missing = allow_missing
        self.exclude_native = exclude_native

    def setup(self) -> None:
        for tool, label in ((paths.VINA, "vina"), (paths.OBABEL, "obabel")):
            if tool == label:  # find_tool fell through to the bare name
                msg = (
                    f"{label} not found on PATH. Set RESCORING_BENCH_"
                    f"{label.upper()} or make it resolvable by "
                    "prolit.external_tools."
                )
                raise FileNotFoundError(msg)

    def score(self, targets: Sequence[str]) -> pd.DataFrame:
        self.setup()
        work = paths.WORK_DIR / "vina"
        dump = work / "pose_scores.csv"
        cli = paths.BENCH_ROOT / "scripts" / "baseline_vina.py"
        cmd = [
            self.python, str(cli),
            "--casf-dir", str(paths.CASF_DIR),
            "--out-csv", str(work / "docking_power.csv"),
            "--rec-dir", str(work / "receptors"),
            "--dump-scores", str(dump),
            "--vina", paths.VINA,
            "--obabel", paths.OBABEL,
            "--workers", str(self.workers),
        ]
        if self.exclude_native:
            cmd.append("--exclude-native")
        # Inherited rather than captured: ~28k poses is hours, and the driver
        # prints a progress bar that is worthless after the fact.
        sys.stderr.flush()
        proc = subprocess.run(cmd, check=False)  # noqa: S603
        if proc.returncode != 0:
            msg = f"baseline_vina failed (rc={proc.returncode}); output is above"
            raise RuntimeError(msg)
        frame = pd.read_csv(dump)
        # The driver has no target filter -- it walks the coreset directory -- so
        # a subset is taken here rather than by not scoring it.
        if targets:
            frame = frame[frame["pdbid"].isin(set(targets))]
        self.check_coverage(frame, targets, self.allow_missing)
        return frame[list(DUMP_COLUMNS)]

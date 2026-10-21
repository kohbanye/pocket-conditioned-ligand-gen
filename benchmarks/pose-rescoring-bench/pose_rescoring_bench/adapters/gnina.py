"""gnina (McNutt et al., J Cheminform 2021) CNN rescoring of CASF decoy poses.

Interaction-featurised in the strongest sense of the table: gnina voxelises the
protein and the ligand together and reads the occupancy of the joint grid, so
every contact is available to it directly.

gnina is a binary, not a library, and its release is built against a newer glibc
than this cluster provides -- the ``.static`` asset still needs ``GLIBC_2.35``
against the host's 2.34. It therefore runs from the official container through
Apptainer, which is also what makes the row reproducible: the image digest pins
the build, where a locally compiled gnina would pin nothing.

``--score_only`` scores the poses exactly as handed over, with no minimisation
and no re-docking, which is the contract every other row in this table is held
to. ``CNNscore`` is what lands in ``head``: it is gnina's pose classifier, and
docking power asks precisely the question it was trained on. ``CNNaffinity`` and
the empirical ``Affinity`` come back in a side file, unused by this table --
picking whichever of the three ranks best would be selection on the test set.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from typing import TYPE_CHECKING

import pandas as pd

from pose_rescoring_bench import paths
from pose_rescoring_bench.adapters.base import DUMP_COLUMNS, PoseBaseline

if TYPE_CHECKING:
    from collections.abc import Sequence


class GninaAdapter(PoseBaseline):
    name = "gnina"
    uses_explicit_contacts = True

    def __init__(
        self,
        cnn_scoring: str = "rescore",
        seed: int = 42,
        gnina: str | None = None,
        allow_missing: Sequence[str] = (),
    ) -> None:
        self.cnn_scoring = cnn_scoring
        self.seed = seed
        self._gnina = gnina
        self.allow_missing = allow_missing

    # -- invocation -------------------------------------------------------
    def command(self) -> str:
        """The shell-quoted command that runs gnina, container included."""
        if self._gnina is not None:
            return self._gnina
        override = os.environ.get("RESCORING_BENCH_GNINA")
        if override:
            return override
        sif = paths.GNINA_SIF
        if sif.exists():
            # --nv exposes the host driver; -B makes the repo visible, since
            # inputs and outputs are addressed by absolute path.
            return shlex.join([
                "apptainer", "exec", "--nv",
                "-B", str(paths.MONOREPO_ROOT),
                str(sif), "gnina",
            ])
        return "gnina"

    def setup(self) -> None:
        paths.require_tool(paths.OBABEL, "obabel")
        cmd = shlex.split(self.command())
        if cmd == ["gnina"]:
            msg = (
                f"gnina not available: no container at {paths.GNINA_SIF} and no "
                "RESCORING_BENCH_GNINA override. Build it with "
                "scripts/fetch_gnina.sh."
            )
            raise FileNotFoundError(msg)

    # -- scoring ----------------------------------------------------------
    def score(self, targets: Sequence[str]) -> pd.DataFrame:
        self.setup()
        work = paths.WORK_DIR / "gnina"
        out = work / "predictions" / f"{self.cnn_scoring}.csv"
        cli = paths.BENCH_ROOT / "scripts" / "gnina_pose_score_cli.py"
        cmd = [
            sys.executable, str(cli),
            "--gnina", self.command(),
            "--casf", str(paths.CASF_DIR),
            "--work", str(work),
            "--cnn-scoring", self.cnn_scoring,
            "--obabel", paths.OBABEL,
            "--seed", str(self.seed),
            "--out", str(out),
        ]
        if targets:
            cmd += ["--pdbids", ",".join(targets)]
        # Inherited rather than captured: 285 targets is hours, and a captured
        # stream only becomes visible once the process ends.
        sys.stderr.flush()
        proc = subprocess.run(cmd, check=False)  # noqa: S603
        if proc.returncode != 0:
            msg = (
                f"gnina_pose_score_cli failed (rc={proc.returncode}); "
                "its output is above"
            )
            raise RuntimeError(msg)
        frame = pd.read_csv(out)
        self.check_coverage(frame, targets, self.allow_missing)
        return frame[list(DUMP_COLUMNS)]

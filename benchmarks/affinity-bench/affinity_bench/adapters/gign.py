"""GIGN (Yang et al., JPCL 2023) via its released code and checkpoints.

Interaction-featurised: every ligand-pocket atom pair within 5 A becomes a graph
edge, so this is not a like-for-like peer of a tokenizer that is handed no
contacts. It is evidence about what explicit interaction modelling buys.

GIGN pins torch 1.10.2 and a torch_sparse built against it, which cannot share
this bench's environment, so scoring runs in ``paths.GIGN_PYTHON`` through
``scripts/gign_score_cli.py``.

The repository ships three checkpoints (repeat0/1/2). All three are scored and
each dump is kept; :meth:`score` returns their mean, because picking the best by
its CASF result would be selection on the test set.
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


class GIGNAdapter(AffinityBaseline):
    name = "gign"
    uses_explicit_contacts = True

    def __init__(self, python: str | None = None, distance: int = 5) -> None:
        self.python = python or paths.GIGN_PYTHON
        # int, not float: GIGN formats this into the cached filenames
        # (`<cid>_{d}A.rdkit`), and its dataset re-derives the same name. 5 and
        # 5.0 produce different names, so a float makes the dataset miss every
        # file the preprocessing just wrote.
        self.distance = distance
        self.member_frames: list[pd.DataFrame] = []

    def setup(self) -> None:
        if not paths.GIGN_REPO.exists():
            msg = (
                f"GIGN submodule missing at {paths.GIGN_REPO}. "
                "Run: git submodule update --init third_party/GIGN"
            )
            raise FileNotFoundError(msg)
        ckpts = paths.gign_checkpoints()
        if not ckpts:
            msg = f"no GIGN checkpoints under {paths.GIGN_MODEL_DIR}"
            raise FileNotFoundError(msg)

    def score(self, complexes: Sequence[Complex]) -> pd.DataFrame:
        self.setup()
        work = paths.WORK_DIR / "gign"
        out_dir = work / "predictions"
        cli = paths.REPO_ROOT / "scripts" / "gign_score_cli.py"
        cmd = [
            self.python, str(cli),
            "--casf", str(paths.CASF_DIR),
            "--work", str(work),
            "--out-dir", str(out_dir),
            "--distance", str(self.distance),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
            msg = f"gign_score_cli failed (rc={proc.returncode})"
            raise RuntimeError(msg)

        members = sorted(out_dir.glob("*.csv"))
        if not members:
            msg = f"gign_score_cli wrote no dumps to {out_dir}"
            raise RuntimeError(msg)
        self.member_frames = [pd.read_csv(p) for p in members]
        for frame in self.member_frames:
            self.check_coverage(frame, complexes)

        # Mean over checkpoints, in pK units (all three predict pK directly, so
        # averaging keeps the scale; a z-sum would not, and the tables report
        # correlations either way).
        merged = self.member_frames[0][["pdbid", "logka", "cluster"]].copy()
        preds = pd.concat(
            [f.set_index("pdbid")["head"] for f in self.member_frames], axis=1
        )
        merged["head"] = merged["pdbid"].map(preds.mean(axis=1))
        return merged[list(DUMP_COLUMNS)]

"""OnionNet-2 (Wang et al., 2021) via its released code and Zenodo model.

Interaction-featurised: the input is a vector of residue-atom contact counts in
62 concentric shells, so like GIGN this is not a like-for-like peer of a
tokenizer that is handed no contacts.

Pins TensorFlow 2.3, which cannot share this bench's environment, so scoring
runs in ``paths.ONIONNET2_PYTHON`` through ``scripts/onionnet2_score_cli.py``.
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

#: An HDF5 file starts with this signature. Upstream tracks the model with
#: git-lfs, so a submodule checkout without git-lfs leaves a small text pointer
#: in its place; handed to Keras that fails deep in the loader with a message
#: that says nothing about what is actually wrong.
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


class OnionNet2Adapter(AffinityBaseline):
    name = "onionnet2"
    uses_explicit_contacts = True

    def __init__(self, python: str | None = None, shells: int = 62) -> None:
        self.python = python or paths.ONIONNET2_PYTHON
        # 62 is upstream's default and the value the released model encodes:
        # 21 residue types x 8 ligand elements x 62 shells = 10,416 = 84 x 124.
        self.shells = shells

    def setup(self) -> None:
        if not paths.ONIONNET2_REPO.exists():
            msg = (
                f"OnionNet-2 submodule missing at {paths.ONIONNET2_REPO}. "
                "Run: git submodule update --init third_party/OnionNet-2"
            )
            raise FileNotFoundError(msg)
        model = paths.ONIONNET2_MODEL
        if not model.exists():
            msg = (
                f"OnionNet-2 model missing at {model}. "
                "Run: python scripts/fetch_weights.py --onionnet2"
            )
            raise FileNotFoundError(msg)
        with model.open("rb") as fh:
            head = fh.read(len(_HDF5_MAGIC))
        if head != _HDF5_MAGIC:
            msg = (
                f"{model} is not an HDF5 file ({model.stat().st_size} bytes). "
                "A git-lfs pointer is the usual cause; re-fetch with "
                "scripts/fetch_weights.py --onionnet2."
            )
            raise ValueError(msg)
        if not paths.ONIONNET2_SCALER.exists():
            msg = f"OnionNet-2 scaler missing at {paths.ONIONNET2_SCALER}"
            raise FileNotFoundError(msg)

    def score(self, complexes: Sequence[Complex]) -> pd.DataFrame:
        self.setup()
        work = paths.WORK_DIR / "onionnet2"
        out = work / "predictions" / "onionnet2.csv"
        cli = paths.REPO_ROOT / "scripts" / "onionnet2_score_cli.py"
        cmd = [
            self.python, str(cli),
            "--casf", str(paths.CASF_DIR),
            "--work", str(work),
            "--model", str(paths.ONIONNET2_MODEL),
            "--scaler", str(paths.ONIONNET2_SCALER),
            "--shells", str(self.shells),
            "--out", str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
            msg = f"onionnet2_score_cli failed (rc={proc.returncode})"
            raise RuntimeError(msg)
        frame = pd.read_csv(out)
        self.check_coverage(frame, complexes)
        return frame[list(DUMP_COLUMNS)]

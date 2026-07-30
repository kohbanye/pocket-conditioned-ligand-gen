"""RTMScore / GenScore CASF docking-power baselines: collect + rerun.

The baselines repo writes one ``<pdbid>_score.dat`` per target (tab-separated
``#code<TAB>score``) under ``casf_work/scores_{rtmscore,genscore}/``. We parse
those into the canonical per-pose schema; RMSD is joined from our own pose dump
at evaluation time (identical decoy set), exactly as the source paper notebook
does.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

_BACKENDS = ("rtmscore", "genscore")


def collect_pose_scores(score_dir: Path) -> pd.DataFrame:
    """Parse ``*_score.dat`` files in ``score_dir`` into (pdbid, pose, native_score)."""
    frames = []
    for f in sorted(score_dir.glob("*_score.dat")):
        t = pd.read_csv(f, sep="\t")
        t.columns = ["pose", "native_score"][: len(t.columns)]
        t["pdbid"] = t["pose"].str.split("_").str[0]
        frames.append(t)
    if not frames:
        msg = f"no *_score.dat files under {score_dir}"
        raise FileNotFoundError(msg)
    return pd.concat(frames, ignore_index=True)[["pdbid", "pose", "native_score"]]


def default_score_dir(baselines_repo: Path, backend: str) -> Path:
    """Location of a backend's per-pose score dir inside the baselines repo."""
    if backend not in _BACKENDS:
        msg = f"backend must be one of {_BACKENDS}, got {backend!r}"
        raise ValueError(msg)
    return baselines_repo / "casf_work" / f"scores_{backend}"


def rerun_docking(
    baselines_repo: Path,
    backend: str,
    extra_args: list[str] | None = None,
) -> None:
    """Re-run a docking-power baseline via ``run_casf.sh`` (micromamba env inside).

    Regenerates ``casf_work/scores_{backend}/*_score.dat`` in the baselines repo,
    which :func:`collect_pose_scores` then parses. GPU + the backend's micromamba
    env are required; intended to run under qsub.
    """
    if backend not in _BACKENDS:
        msg = f"backend must be one of {_BACKENDS}, got {backend!r}"
        raise ValueError(msg)
    script = baselines_repo / "run_casf.sh"
    cmd = ["bash", str(script), backend, *(extra_args or [])]
    subprocess.run(cmd, check=True, cwd=str(baselines_repo))  # noqa: S603

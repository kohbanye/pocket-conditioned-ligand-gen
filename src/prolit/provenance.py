"""Record what produced a run, next to what the run produced.

Weights, caches and dumps are not in git, and the job scripts that launched them
are not either. What is missing is the link between them: given a checkpoint,
which command, which code, which seed made it?

That link does not belong in git. Git has the code; the run directory has the
weights; the only thing that has to be written down is which pair went together,
and it belongs beside the weights so the two travel as one. A run that gets
deleted takes its provenance with it, which is correct -- they are the same
thing.

So every run writes ``run.json`` into its own directory:

    {"command": ["pipelines/train/clm.py", "--seed", "7", ...],
     "git": {"sha": "3ba7053", "dirty": false},
     "seed": 7, "hostname": "r9n2", "started": "2026-07-31T...",
     "job": {"name": "lm_pre", "resource": "node_f", "hours": 8}}

``job`` is present when the run came from ``jobs/submit.py``, which passes it
through the environment -- the submitter cannot know the run directory (with no
``--run-name`` it is the W&B run id, decided at startup), so the run records it
instead. For one arm of a ``--sweep`` it also carries
``"sweep": {"listwise-tau": "0.5"}``: which comparison this run belongs to,
which the command alone does not say.

The dirty flag matters more than the SHA. A clean SHA means the code is
recoverable; a dirty one means it is not, and a number produced from it cannot be
reproduced from git alone.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightning as L

MANIFEST_NAME = "run.json"

#: Environment variables ``jobs/submit.py`` sets so a run can record how it was
#: submitted. Absent for an interactive run, which is fine -- the command is
#: recorded either way.
_JOB_ENV = {
    "name": "PROLIT_JOB_NAME",
    "resource": "PROLIT_JOB_RESOURCE",
    "hours": "PROLIT_JOB_HOURS",
}

#: Set by ``--sweep``, holding this run's point as JSON. The values are in the
#: command already; what this adds is that the run is one arm of a comparison,
#: and which axis was varied -- the thing ``job_pose_v8`` / ``_v10`` filenames
#: used to encode.
_SWEEP_ENV = "PROLIT_JOB_SWEEP"


def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=repo, capture_output=True, text=True, check=True, timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip()


def git_state(repo: Path | None = None) -> dict[str, Any]:
    """Current commit and whether the tree has uncommitted changes.

    Returns ``{"sha": None}`` outside a git checkout rather than raising: a run
    that happens elsewhere should still record everything else it can.
    """
    repo = repo or Path(__file__).resolve().parents[2]
    sha = _git(repo, "rev-parse", "HEAD")
    if sha is None:
        return {"sha": None, "dirty": None, "branch": None}
    status = _git(repo, "status", "--porcelain")
    return {
        "sha": sha,
        "dirty": bool(status),
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
    }


def _job_env() -> dict[str, Any] | None:
    found: dict[str, Any] = {
        k: os.environ[v] for k, v in _JOB_ENV.items() if v in os.environ
    }
    sweep = os.environ.get(_SWEEP_ENV)
    if sweep:
        try:
            found["sweep"] = json.loads(sweep)
        except json.JSONDecodeError:
            # Recording it wrongly beats recording nothing; the run still ran.
            found["sweep"] = sweep
    if not found:
        return None
    job_id = os.environ.get("JOB_ID") or os.environ.get("SLURM_JOB_ID")
    if job_id:
        found["id"] = job_id
    return found


def run_manifest(
    *,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the record for the currently-running process."""
    manifest: dict[str, Any] = {
        "command": [sys.argv[0], *sys.argv[1:]],
        "git": git_state(),
        "seed": seed,
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
    }
    try:
        import torch  # noqa: PLC0415

        manifest["torch"] = torch.__version__
        manifest["cuda"] = torch.version.cuda
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass
    job = _job_env()
    if job:
        manifest["job"] = job
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(
    run_dir: str | Path,
    *,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``run.json`` into ``run_dir`` and return its path.

    Overwrites: a resumed or re-run job should describe how it was *last* run,
    not the first time. The previous manifest is not worth keeping -- the
    checkpoint it described has been overwritten too.
    """
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / MANIFEST_NAME
    target.write_text(json.dumps(run_manifest(seed=seed, extra=extra), indent=2) + "\n")
    return target


def read_manifest(run_dir: str | Path) -> dict[str, Any] | None:
    """Read a run's manifest, or None if it predates this or was never written."""
    path = Path(run_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


class RecordProvenance(L.Callback):
    """Lightning callback that writes ``run.json`` beside the checkpoints.

    Deliberately fires at fit start rather than at submit time: without
    ``--run-name`` the checkpoint directory is the W&B run id, which does not
    exist until the logger has initialised. By then Lightning knows where
    checkpoints will land, and that is the directory the manifest belongs in.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.written: Path | None = None

    def _resolve_dir(self, trainer: Any) -> Path | None:  # noqa: ANN401
        callback = getattr(trainer, "checkpoint_callback", None)
        dirpath = getattr(callback, "dirpath", None)
        if dirpath:
            return Path(dirpath)
        log_dir = getattr(trainer, "log_dir", None)
        return Path(log_dir) if log_dir else None

    def setup(self, trainer: Any, pl_module: Any, stage: str) -> None:  # noqa: ANN401, ARG002
        """Write once, on the rank-0 process, at the start of fit."""
        if stage != "fit" or getattr(trainer, "global_rank", 0) != 0:
            return
        run_dir = self._resolve_dir(trainer)
        if run_dir is None:
            return
        self.written = write_manifest(run_dir, seed=self.seed)

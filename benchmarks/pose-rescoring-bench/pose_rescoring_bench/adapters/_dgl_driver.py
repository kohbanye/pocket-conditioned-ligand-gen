"""Shared subprocess plumbing for the two Shen-lab pose scorers.

RTMScore and GenScore come from the same group and expose the same three entry
points -- ``VSDataset``, ``extract_pocket``, ``run_an_eval_epoch`` -- so one
driver script scores both and the adapters differ only in which repository,
interpreter and checkpoint they name. Keeping that in one place is also what
keeps the *protocol* identical between them, which is the only reason their two
rows belong in the same table.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from pose_rescoring_bench import paths

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class Backend:
    """Everything that distinguishes one of the two scorers from the other."""

    backend: str
    python: str
    repo: Path
    checkpoint: str
    #: GenScore only: the graph encoder, which has to match the checkpoint.
    encoder: str | None = None
    num_workers: int = 4


def run_pose_cli(
    spec: Backend,
    targets: Sequence[str],
    out: Path,
) -> pd.DataFrame:
    """Score ``targets`` with one backend and read back the per-pose dump.

    The driver's output goes straight to ours rather than being captured:
    these runs take hours over 285 targets, and a captured
    stream is invisible until the process ends -- which, for a job, means the
    only progress report arrives with the result. Inheriting the streams puts
    each target's line in the job log as it happens, and puts the traceback
    there too, so the failure path loses nothing.
    """
    work = paths.WORK_DIR / spec.backend
    cli = paths.BENCH_ROOT / "scripts" / "casf_pose_score_cli.py"
    cmd = [
        spec.python, str(cli),
        "--backend", spec.backend,
        "--repo", str(spec.repo),
        "--casf", str(paths.CASF_DIR),
        "--work", str(work),
        "--checkpoint", spec.checkpoint,
        "--num-workers", str(spec.num_workers),
        "--obabel", paths.OBABEL,
        "--out", str(out),
    ]
    if spec.encoder is not None:
        cmd += ["--encoder", spec.encoder]
    if targets:
        cmd += ["--pdbids", ",".join(targets)]
    sys.stderr.flush()
    proc = subprocess.run(cmd, check=False, env=_child_env(spec.python))  # noqa: S603
    # A non-zero exit means at least one target failed, and the driver still
    # wrote the rest. Returning the partial dump lets the coverage check name
    # what is missing, which is the useful error; discarding 284 scored targets
    # because the 285th could not be encoded is not.
    if not out.exists():
        msg = (
            f"casf_pose_score_cli ({spec.backend}) failed "
            f"(rc={proc.returncode}) without writing {out}; its output is above"
        )
        raise RuntimeError(msg)
    return pd.read_csv(out)


def _child_env(python: str) -> dict[str, str]:
    """The environment a conda-built scorer needs to import at all.

    Both environments ship their own ``libstdc++``/``libgfortran``, and scipy
    and ProDy fail to import against the system ones. ``run_casf.sh`` set
    ``LD_LIBRARY_PATH`` by hand; deriving it from the interpreter's own prefix
    means an adapter works wherever its env happens to live. ``DGLBACKEND`` is
    likewise what upstream's launcher exports.
    """
    env = dict(os.environ)
    prefix = Path(python).resolve().parent.parent
    lib = prefix / "lib"
    if lib.is_dir():
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib}:{existing}" if existing else str(lib)
        babel = lib / "openbabel" / "3.1.0"
        if babel.is_dir():
            env.setdefault("BABEL_LIBDIR", str(babel))
    env.setdefault("DGLBACKEND", "pytorch")
    return env


def require_repo(repo: Path, submodule: str, checkpoint: str) -> None:
    """Fail with the command that fixes it, rather than deep inside a loader."""
    # The driver converts every decoy stack with OpenBabel before it scores it.
    paths.require_tool(paths.OBABEL, "obabel")
    if not repo.exists():
        msg = (
            f"{submodule} submodule missing at {repo}. "
            f"Run: git submodule update --init third_party/{submodule}"
        )
        raise FileNotFoundError(msg)
    ckpt = repo / "trained_models" / checkpoint
    if not ckpt.exists():
        msg = f"{submodule} checkpoint missing: {ckpt}"
        raise FileNotFoundError(msg)

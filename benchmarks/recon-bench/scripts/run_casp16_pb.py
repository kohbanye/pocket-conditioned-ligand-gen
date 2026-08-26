"""Run the CASP16 reconstruction + PoseBusters sweep under this bench's own venv.

Two things stop the sweep from being a plain ``jobs/submit.py`` command:

* recon-bench has its own interpreter (ESM3 needs a forked ``transformers``),
  while a generated job script runs everything through the source repo's ``$PY``;
* the CASP16 archives live in the pre-split repository, not under this bench, so
  ``RECON_BENCH_DATA_DIR`` has to be set before the runner imports ``paths``.

Both are handled here, so the job script only has to name this file. Launched
from the batch queue rather than interactively: the previous attempt ran on the
interactive node and was reaped after 21 of 303 complexes, leaving a log whose
last line looked like ordinary progress.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent
DEFAULT_DATA = Path(
    "/gs/bs/tga-ohuelab/sakano/git/protein-ligand-3d-reconstruction-bench/data"
)


def main() -> int:
    arms = sys.argv[1:] or ["joint_bond", "joint_noleak", "joint", "separate"]
    python = BENCH / ".venv" / "bin" / "python"
    if not python.exists():
        msg = f"recon-bench venv missing: {python}. Run `uv sync` in {BENCH}."
        raise SystemExit(msg)

    env = dict(os.environ)
    env.setdefault("RECON_BENCH_DATA_DIR", str(DEFAULT_DATA))
    data = Path(env["RECON_BENCH_DATA_DIR"]) / "casp16" / "index.json"
    if not data.exists():
        msg = f"CASP16 index missing: {data}. Set RECON_BENCH_DATA_DIR."
        raise SystemExit(msg)

    cmd = [
        str(python),
        "scripts/run_reconstruction.py",
        "--models", "own_allatom",
        "--dataset", "casp16",
        "--pb-valid",
        "--allatom-arms", *arms,
    ]
    print("running:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(BENCH), env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

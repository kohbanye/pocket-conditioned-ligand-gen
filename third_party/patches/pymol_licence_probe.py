"""Stop PyMOL aborting the interpreter where /opt/schrodinger is root-only.

``pymol/licensing.py`` appends ``/opt/schrodinger/licenses`` to the licence
search path unconditionally on Linux -- ``SCHRODINGER`` is not consulted. When
that directory exists but is not readable, the C++ side stats it, throws
``std::filesystem_error`` and aborts: no Python traceback, just a core dump, at
``pymol2.PyMOL().start()``. FLOWR starts a session while importing
``flowr.gen``, so this made FLOWR unrunnable here for reasons that looked like
a CUDA or checkpoint problem.

Nothing in this benchmark uses a licensed PyMOL feature. Guard the append.

    python third_party/patches/pymol_licence_probe.py \
        /path/to/env/lib/python3.11/site-packages/pymol/licensing.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ANCHOR = "        path.append(u'/opt/schrodinger/licenses')"
GUARDED = (
    "        # Patched: PyMOL appends this unconditionally, and where\n"
    "        # /opt/schrodinger exists but is root-only the C++ side stats it,\n"
    "        # throws std::filesystem_error and aborts the interpreter -- no\n"
    "        # traceback, just a core dump. Nothing here uses a licensed feature.\n"
    "        if os.access(u'/opt/schrodinger/licenses', os.R_OK):\n"
    "            path.append(u'/opt/schrodinger/licenses')"
)


def main() -> None:
    target = Path(sys.argv[1])
    source = target.read_text()
    if GUARDED.splitlines()[-1] in source:
        print(f"already patched: {target}")
        return
    if ANCHOR not in source:
        msg = f"anchor not found in {target}; PyMOL may have changed the probe"
        raise SystemExit(msg)
    shutil.copy2(target, target.with_suffix(".py.orig"))
    target.write_text(source.replace(ANCHOR, GUARDED))
    print(f"patched: {target}")


if __name__ == "__main__":
    main()

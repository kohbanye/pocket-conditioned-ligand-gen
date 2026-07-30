"""Locate the external binaries the evaluation paths shell out to.

Docking and ligand preparation are done by programs that are not Python
packages and cannot be pinned in the lockfile: AutoDock Vina, Open Babel, and
ADFRsuite's ``prepare_receptor``. Where they live is a property of the machine,
so this resolves them at call time instead of baking one person's install
prefix into the source.

Resolution order, per tool:

1. its environment variable (``PROLIT_VINA``, ``PROLIT_OBABEL``,
   ``PROLIT_PREPARE_RECEPTOR``) — set this when the binary is not on ``PATH``
   or when a specific build is required;
2. ``PATH``.

If neither finds it, :func:`require_tool` raises with the variable to set,
rather than letting a subprocess fail later with a bare "No such file".
"""

from __future__ import annotations

import os
import shutil

#: tool name -> (environment variable, what it is used for)
_TOOLS: dict[str, tuple[str, str]] = {
    "vina": ("PROLIT_VINA", "AutoDock Vina — docking and rescoring"),
    "obabel": (
        "PROLIT_OBABEL",
        "Open Babel — SDF/PDBQT conversion and bond perception",
    ),
    "prepare_receptor": (
        "PROLIT_PREPARE_RECEPTOR",
        "ADFRsuite prepare_receptor — receptor PDB -> PDBQT",
    ),
}


def find_tool(name: str) -> str | None:
    """Return the path to ``name``, or None if it cannot be found."""
    env_var, _ = _TOOLS.get(name, (f"PROLIT_{name.upper()}", ""))
    override = os.environ.get(env_var)
    if override:
        return override
    return shutil.which(name)


def require_tool(name: str) -> str:
    """Return the path to ``name``, or raise explaining how to point at it."""
    found = find_tool(name)
    if found:
        return found
    env_var, purpose = _TOOLS.get(name, (f"PROLIT_{name.upper()}", ""))
    detail = f" ({purpose})" if purpose else ""
    msg = (
        f"{name!r} not found on PATH{detail}. "
        f"Install it, or set {env_var} to its full path."
    )
    raise FileNotFoundError(msg)


def tool_default(name: str) -> str:
    """Best guess for an argparse default: a resolved path, else the bare name.

    Returning the bare name keeps ``--help`` readable and lets the failure
    surface at :func:`require_tool` time with a useful message, rather than
    making the parser fail on a machine that simply has the tool elsewhere.
    """
    return find_tool(name) or name

"""The FoldToken CLI must import FoldToken's chroma, not ProLIT's package.

This is a regression test for a silent, total baseline failure. FoldToken
vendors a package literally named ``src``, and its reconstruction CLI does
``from src.chroma.data import Protein`` after putting the FoldToken checkout on
``sys.path``. A repo-wide ``src`` -> ``prolit`` rename rewrote that import too,
which made ``tests/test_layering.py`` pass -- it listed ``src`` as one of our
own import roots and so read the third-party import as a layer violation.

Nothing raised at import time, because the import lives inside ``main()``. Every
FoldToken sample simply came back ``setup failed: ModuleNotFoundError`` -- a
message that reads like a missing checkpoint, and one the results table shows by
omitting the model's rows rather than by failing. The baseline was dead in every
run until someone counted the rows.

The layering test no longer lists ``src``. This one states the requirement
positively, so the next rename has to argue with a test that says what it wants.
"""

from __future__ import annotations

import ast
from pathlib import Path

CLI = Path(__file__).resolve().parents[1] / "scripts" / "foldtoken_reconstruct_cli.py"


def _import_from_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
    }


def test_chroma_comes_from_foldtokens_own_src_package() -> None:
    modules = _import_from_modules(CLI)
    assert "src.chroma.data" in modules, (
        f"{CLI.name} must import `from src.chroma.data import Protein` -- `src` is "
        f"FoldToken's vendored package, reachable because main() puts the FoldToken "
        f"checkout on sys.path. Found: {sorted(modules)}"
    )
    assert not any(m.startswith("prolit.chroma") for m in modules), (
        "ProLIT has no `chroma` subpackage; this import was rewritten by a "
        "repo-wide src->prolit rename and it disabled the whole FoldToken baseline."
    )

"""The dependency direction between top-level directories, enforced.

Four layers, and edges only ever point downward:

    pipelines/  ──┐
    benchmarks/ ──┼──>  src/prolit/   (the library: no argparse, no I/O policy)
    scripts/    ──┘

* **prolit** knows nothing about the layers above it. It cannot import them and
  cannot know a corpus, a benchmark or a CLI exists.
* **pipelines**, **benchmarks** and **scripts** are siblings. None imports
  another. When two of them need the same thing, that thing belongs in prolit --
  which is how ``PoseEncoder``, ``parse_mol2_multi`` and the Vina helpers got
  there, after living in an eval script that a corpus builder imported from and
  a benchmark had copied.

Subprocess calls are a dependency too, and are checked the same way: a benchmark
may shell into ``scripts/`` (that is the point of scripts/ -- it is the surface
benchmarks drive), but nothing may shell into a benchmark from below.

This test exists because those edges are invisible in review. Every violation it
lists was a real one before it was written.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Directory -> the top-level packages its files may import, on top of whatever
#: package the file itself belongs to (a layer may always import itself).
ALLOWED_IMPORTS: dict[str, set[str]] = {
    "src/prolit": set(),  # the library depends on nothing of ours
    "pipelines": {"prolit"},
    "scripts": {"prolit"},
    # Each benchmark may use the library and the shared bench package; the
    # per-benchmark packages are added below, since a benchmark importing a
    # *different* benchmark is exactly what this forbids.
    "benchmarks": {"prolit", "prolit_bench"},
}

#: Our own top-level import roots, so third-party imports are ignored.
OURS = {
    "prolit", "prolit_bench", "pipelines", "scripts", "benchmarks", "src",
    "recon_bench", "pose_rescoring_bench", "sbdd_bench",
}


def _tracked(prefix: str) -> list[Path]:
    out = subprocess.run(  # noqa: S603
        ["git", "ls-files", "-z", prefix],  # noqa: S607
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.split("\0")
    return [REPO / f for f in out if f.endswith(".py")]


def _imported_roots(path: Path) -> set[str]:
    """Top-level module names this file imports, ours only."""
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots & OURS


def _own_package(rel: Path, layer: str) -> set[str]:
    """The package a file belongs to, which it may always import.

    Within a layer, siblings importing each other is normal -- what the rule
    forbids is reaching into a *different* layer.
    """
    if layer == "src/prolit":
        return {"prolit"}
    if layer in ("pipelines", "scripts"):
        return {layer}
    if layer == "benchmarks" and len(rel.parts) > 2:
        # benchmarks/<bench-dir>/<package>/... -- the package, not the directory.
        # A bench either holds its package directly or under src/.
        bench = REPO / rel.parts[0] / rel.parts[1]
        roots = [bench, bench / "src"]
        return {
            d.name
            for root in roots
            if root.is_dir()
            for d in root.iterdir()
            if d.is_dir() and (d / "__init__.py").exists()
        }
    return set()


def test_no_upward_or_sideways_imports() -> None:
    """Each layer may only import what ALLOWED_IMPORTS grants it."""
    violations: list[str] = []
    for layer, allowed in ALLOWED_IMPORTS.items():
        for path in _tracked(layer):
            rel = path.relative_to(REPO)
            permitted = allowed | _own_package(rel, layer)
            violations.extend(
                f"{rel} imports {root!r} (allowed: {sorted(permitted)})"
                for root in _imported_roots(path) - permitted
            )
    assert not violations, "layer violations:\n  " + "\n  ".join(sorted(violations))


def test_library_never_imports_the_layers_above_it() -> None:
    """The strictest edge, stated separately because it is the one that matters.

    If prolit reaches up into pipelines/, scripts/ or benchmarks/, it stops
    being a library and the whole structure collapses into one mutual blob.
    """
    forbidden = {"pipelines", "scripts", "benchmarks", "prolit_bench"}
    offenders = [
        f"{p.relative_to(REPO)} -> {sorted(_imported_roots(p) & forbidden)}"
        for p in _tracked("src/prolit")
        if _imported_roots(p) & forbidden
    ]
    assert not offenders, "prolit must not import upward:\n  " + "\n  ".join(offenders)


def test_nothing_below_shells_into_a_benchmark() -> None:
    """Subprocess edges follow the same direction as imports.

    Benchmarks drive ``scripts/`` and ``pipelines/`` as subprocesses; the
    reverse would make a corpus builder depend on an evaluation harness.
    """
    offenders = []
    for layer in ("src/prolit", "pipelines", "scripts"):
        for path in _tracked(layer):
            text = path.read_text()
            if "benchmarks/" in text and "noqa: layering" not in text:
                offenders.append(str(path.relative_to(REPO)))
    assert not offenders, (
        "these reference benchmarks/ from below it:\n  "
        + "\n  ".join(sorted(offenders))
    )


def test_scripts_stays_small() -> None:
    """scripts/ is the benchmark-facing surface, not a place to put things.

    It grew to 62 files once. The bound is deliberately tight so adding to
    it is a decision rather than a habit: anything reusable belongs in prolit,
    anything that builds a corpus or trains belongs in pipelines/.
    """
    limit = 12
    entries = sorted(p.name for p in _tracked("scripts"))
    assert len(entries) <= limit, (
        f"scripts/ has {len(entries)} python entry points (limit {limit}): {entries}"
    )

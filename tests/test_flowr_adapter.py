"""The FLOWR arm is a comparison, so its wiring has to fail loudly.

Two failures cost hours in this repository and neither announced itself:

- PyMOL appends ``/opt/schrodinger/licenses`` to its licence search path
  unconditionally on Linux. Where that directory exists but is root-only the
  C++ side aborts the interpreter -- a core dump with no Python traceback,
  raised while importing ``flowr.gen``. It looked like a CUDA or checkpoint
  problem for a long time.
- FLOWR's own diversity filter fingerprints every sample, including ones that
  failed to build an RDKit mol, and dies on the ``None``. It only appears at
  larger sample counts (10 was fine, 100 was not).

Neither can be exercised without FLOWR's environment, so these tests check the
things that *are* checkable: that the patch is recorded, that the adapter is
registered, and that it refuses clearly when its weights are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
# The benchmark is a sibling package with its own root; tests import it the way
# its own scripts do.
sys.path.insert(0, str(REPO / "benchmarks/sbdd-bench"))

from sbdd_bench import adapters  # noqa: E402
from sbdd_bench.adapters.flowr import FlowrAdapter  # noqa: E402


def test_the_pymol_licence_patch_is_recorded() -> None:
    p = REPO / "third_party/patches/pymol_licence_probe.py"
    assert p.exists(), "the PyMOL patch must stay in the repo to be reappliable"
    text = p.read_text()
    assert "opt/schrodinger/licenses" in text
    assert "os.access" in text, "the guard is the whole point of the patch"


def test_the_none_ligand_patch_is_recorded() -> None:
    p = REPO / "third_party/patches/flowr_none_ligands.patch"
    assert p.exists()
    assert "filter_diverse_ligands_bulk" in p.read_text()


def test_flowr_is_a_registered_arm() -> None:
    assert "flowr" in adapters.available()


def test_it_names_the_missing_checkpoint() -> None:
    """A missing weight file must say which file and where to get it."""
    a = FlowrAdapter(ckpt="/nonexistent/flowr_noHs.ckpt")
    with pytest.raises(FileNotFoundError, match=r"flowr_noHs\.ckpt"):
        a.setup()

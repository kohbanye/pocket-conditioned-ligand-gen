"""sbdd-bench: a shared, model-agnostic evaluation suite for SBDD 3D ligand
generators.

The package separates the two halves of an SBDD benchmark:

* **Generation** (``sbdd_bench.adapters``) drives each model — DiffSBDD,
  TargetDiff, DiffGui, and the in-house pocket-conditioned-ligand-gen — in its
  own environment to produce a standard ``generated.sdf`` per target.
* **Evaluation** (``sbdd_bench.metrics`` and the modules it orchestrates) reads
  those SDFs and scores every model identically: chemical validity, docking
  (Vina Score / Min / Dock), pose quality (PoseBusters, clashes, strain),
  diversity / novelty, and a composite hit-rate.
"""

from __future__ import annotations

__all__ = ["paths", "types"]

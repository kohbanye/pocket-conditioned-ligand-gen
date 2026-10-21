"""Baseline collection from a sibling checkout. Superseded by ``adapters``.

**New work belongs in ``pose_rescoring_bench.adapters``**, which runs each
baseline here, from a pinned ``third_party/`` submodule, and writes the same
per-pose dump. This module is kept because the numbers currently published were
computed from what it collects, and re-measuring a published row is a decision
about the paper rather than a side effect of a refactor. It cannot be re-run
anywhere the sibling checkout is absent; an adapter can.


The baseline methods themselves (RTMScore, GenScore, Vina, Boltz-2, and the SBDD
generators DiffGui/TargetDiff/DiffSBDD) live in the sibling ``baselines/`` and
``sbdd-bench/`` repos with their own micromamba environments and weights. These
modules do two things:

1. **collect** — parse those repos' existing per-sample outputs into this repo's
   canonical dump schema (used to seed ``results/`` with known-reproducible
   numbers without re-running anything);
2. **rerun** — thin subprocess wrappers that re-run the baseline under the exact
   same protocol to regenerate those outputs when needed.
"""

"""Baseline reproduction code (lives here, per the "no models but code" split).

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

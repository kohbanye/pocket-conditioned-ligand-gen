# prolit-bench-common

The parts the three benchmarks must not disagree about.

- **`variants.py`** — which weights each tokenizer arm means. Read its module
  docstring first: the benchmarks had silently drifted apart here, and the
  discrepancy is recorded rather than papered over.
- **`stats.py`** — paired t-test, Steiger, Wilcoxon, McNemar, bootstrap CIs and
  Holm correction. Every table in the paper that claims a difference is or is
  not significant goes through this.

Kept dependency-light on purpose (numpy / pandas / scipy, no torch, no RDKit):
`recon-bench` cannot share the main environment, so anything heavy here would have
to be installed twice.

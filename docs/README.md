# docs/

```
results/   frozen records: which checkpoints and hyperparameters produced which number
notes/     dated investigation logs — what was tried, what happened, why it was dropped
figures/   figure sources for the paper
```

`results/` is the authority when a number is in question. `best_allatom_configs.md`
is the reproducibility manifest for the paper's three tasks: checkpoint paths,
hyperparameters, the exact eval command, and the metric each produced.

`notes/` is history, not documentation. The entries are accurate as of their
date and are deliberately not updated — several describe approaches that were
later abandoned, and knowing what failed is most of their value. Do not treat a
path or flag mentioned there as current without checking.

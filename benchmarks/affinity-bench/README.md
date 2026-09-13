# prolit-affinity-bench

**Question**: how tightly does this ligand bind this pocket, and does ProLIT
answer it better than the methods built for it?

CASF-2016, 285 core complexes in 57 five-ligand clusters, crystal poses:

- **scoring power** — Pearson R between predicted and measured pK, all 285;
- **ranking power** — mean within-cluster Spearman rho over the 57 clusters.

Against **GenScore**, **Boltz-2** and **Vina**, on the same poses, judged by
significance rather than by the point estimate (Steiger for two dependent
correlations, Wilcoxon over per-cluster rho, Holm across the set).

## Why this is its own benchmark

It was a field inside `pose-rescoring-bench`, sharing that bench's arm registry.
The two tasks share an architecture and a tokenizer but nothing else: different
corpus, different label, different backbone (`wxlhgqx3` here against
`j90rlrgm` there), and each head has been measured to carry none of the other's
signal — the pose head scores affinity at R = −0.036. One registry described two
models under one name, which is the drift `prolit_bench.variants` exists to
prevent. Tokenizer *arm identity* is still defined once, there; which head and
backbone an affinity arm means is defined once, here, in `variants.py`.

## Layout

```
affinity_bench/
  config.py      paths + how a complex is scored (n_frames)
  variants.py    which tokenizer / MLM / head each arm means
  inference.py   CASF crystal complexes -> per-complex dump (GPU)
  metrics.py     scoring R, ranking rho, per-cluster rho, z-sum
  aggregate.py   metrics + significance against a named reference
  report.py      the dump tree -> tables
  baselines.py   GenScore / Vina / Boltz-2 collectors
scripts/         argparse entry points
results/<arm>/<head>.csv     our dumps, one CSV per head
results/{genscore,boltz2,vina}/scoring.csv
```

Every CSV inside one arm directory is a member of that arm's z-sum ensemble, so
a second protocol of the same arm goes in its own directory (`--suffix`), not a
second file.

## Analysis (no GPU)

```sh
uv run python scripts/collect_baselines.py   # seed results/ from the sibling repos
uv run python scripts/make_tables.py         # tables -> results/tables/
uv run pytest
```

`make_tables.py` discovers arm directories, so a head trained overnight shows up
without an edit. It prints a **scoreboard** next to the table: the target is to
beat GenScore on *both* metrics, and the scoreboard says whether that happened,
so a night's work is read against the target that was set rather than against
whichever metric moved.

## Inference (GPU, qsub)

```sh
.venv/bin/python scripts/infer_affinity.py --arm e250_kdki_mean --n-frames 16
```

`--n-frames` averages the prediction over that many random rigid rotations of
the complex. The pocket-canonical frame comes from the pocket, so a complex's
affinity should not depend on it; quantization makes it depend on it anyway, and
that spread is uncorrelated with the pK while the signal is not, so it pulls
both metrics toward zero. The pose head needed 16 draws before the spread
stopped falling.

## Reading the tables

`make_tables.py` prints the seed tables first and labels the per-arm table
`n=1 each [a single draw, not a measurement]`. That ordering is deliberate.
Retraining one of these heads with nothing changed but the seed moves scoring R
by about 0.05 and ranking rho by about 0.04 -- larger than any recipe difference
measured so far -- so a row in the per-arm table is a draw from a distribution,
not a property of the arm. Four conclusions were drawn from single runs on
2026-09-13 and three of them had to be retracted.

Compare arms **paired on the seed** (`aggregate.paired_by_seed`); the verdict is
`n_better` out of `n_seeds`, because a three-seed p-value has almost no power
and a 2-of-3 split is the signature of nothing being there.

## Status

Against GenScore's **R 0.816 / rho 0.735**:

| | scoring R | ranking rho |
|---|---|---|
| `e250_rot1`, 3 seeds | 0.754 +/- 0.051 | 0.695 +/- 0.043 |
| the same three runs, predictions averaged | 0.784 | **0.735** |

Averaging the seeds is a diagnostic, not a reported arm -- but it says the
systematic ranking ability is already GenScore's, and that a single run's rho is
mostly reading training noise. The scoring gap is real and is a size effect:
`corr(heavy atoms, prediction)` is 0.665 for us against 0.500 for the truth and
0.415 for GenScore, and optimally reweighting our size and size-free components
buys +0.0005, so it is missing signal rather than mis-calibration.

Full diagnosis: `docs/results/2026-09-13_where_the_affinity_head_loses.md`.
What is being tried, and what would count as it having failed:
`docs/notes/2026-09-13_affinity_prereg.md`.

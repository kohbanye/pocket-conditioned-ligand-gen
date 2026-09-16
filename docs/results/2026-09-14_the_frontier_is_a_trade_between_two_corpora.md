# Ensembling reaches the frontier, and the frontier is a trade

2026-09-14, evening. With ensembles permitted, every registered family was
z-summed over **all** of its seeds -- no member chosen by score -- and scored on
CASF at matched 285-complex coverage against GenScore 0.8127 / 0.7281.

## Single-family ensembles

| family (all seeds) | R | rho |
|---|---:|---:|
| `e250_union_noconf` (3) | 0.8092 (−0.0035) | 0.7000 (−0.0281) |
| `e250_union_clean` (3) | 0.8075 (−0.0052) | 0.7105 (−0.0175) |
| `e250_rot1` (3) | 0.7827 (−0.0300) | **0.7386 (+0.0105)** |

`e250_rot1`'s ensemble is **the first thing in this project to exceed GenScore
on ranking rho**. Its scoring R is 0.030 short.

Ensembling itself is worth a lot: +0.010 to +0.029 R and +0.004 to +0.047 rho
over the mean single seed of the same family. None of that required a new
mechanism -- it is variance reduction on a metric whose seed spread is large.

## Cross-corpus ensembles: the trade does not dissolve

| ensemble | members | R | rho |
|---|---:|---:|---:|
| `union_noconf` + `union_clean` | 6 | **0.8129 (+0.0003)** | 0.7140 (−0.0140) |
| `rot1` + `union_clean` | 6 | 0.8050 (−0.0077) | **0.7351 (+0.0070)** |
| `rot1` + `union_noconf` | 6 | 0.8061 (−0.0065) | 0.7281 (±0.0000) |
| **all registered families** | **39** | **0.8107 (−0.0019)** | **0.7228 (−0.0053)** |

Either metric can be beaten; neither combination beats both. Mixing rot1 in
buys rho and costs R, monotonically. **The all-families ensemble involves no
selection at all and lands −0.0019 / −0.0053** -- the honest number, and the
closest anything has come. The two rows that do beat a metric were identified by
looking at CASF, so they are exploratory, not results.

## Why the two corpora carry different metrics

**Scoring R is a size story, as documented.** Correlation of the prediction with
ligand heavy-atom count:

| | corr(heavy, prediction) |
|---|---:|
| `rot1` ensemble | +0.6667 |
| `union_noconf` ensemble | +0.5873 |
| **true pK** | **+0.5005** |
| GenScore | +0.4149 |

Adding PDBbind moves the model's size reliance from +0.667 toward the true
+0.500, and that is where its R advantage comes from. GenScore sits on the other
side, *under*-using size, and still scores higher -- so proximity to the true
size coupling is not itself the objective.

**Ranking rho is not a size story, and not a few bad clusters.** Over 57
clusters: rot1 beats GenScore on 17, union on 18, both beat it on 16, both lose
on 19. Only **2 of 57** clusters lose by more than 0.3 rho, and they account for
1.10 of the 4.00 total deficit. The remaining ~73% is an accumulation of small
per-cluster losses with no concentration to attack.

That rules out the shape of fix that would have been cheapest -- there is no
handful of pathological targets whose repair moves the mean.

## What this implies for the next step

The gap is now small enough that variance reduction alone is a credible route:
the zero-selection ensemble is 0.002 and 0.005 away. Ensembling gain grows with
member count, and the two strongest families have only three seeds each. Adding
seeds is simple, changes no architecture, keeps the head and the tokenizer
fixed, and is not a mechanism claim.

Weight-fitting was considered and rejected for now: the only non-CASF dumps are
three `rot1` runs on the PDBbind holdout, whose clusters are singletons, so rho
is undefined there and no GenScore dump exists on that set. Fitting weights on
CASF would be selection on the test set.

---

## CORRECTION (same day): the rho win did not survive more seeds

This note recorded `e250_rot1`'s 3-seed ensemble at rho **0.7386** and called it
"the first thing in this project to exceed GenScore on ranking rho". Both
families were then extended to six seeds. It does not hold.

| family | R (3 -> 6 seeds) | rho (3 -> 6 seeds) |
|---|---|---|
| `e250_rot1` | 0.7827 -> 0.7913 | **0.7386 -> 0.6947** |
| `e250_union_noconf` | 0.8092 -> 0.8107 | **0.7000 -> 0.6737** |
| all-families | 0.8107 -> 0.8107 | 0.7175 -> 0.7175 |

Doubling the members bought +0.009 and +0.002 on R and **cost 0.044 and 0.026
on rho**. The rot1 rho figure was a three-seed draw, not a property of the
family, and at six seeds it sits at 0.6947 -- below GenScore's 0.7281.

So the closing paragraph of this note is withdrawn: variance reduction is not a
route to the target. It was the explicit hypothesis, it was tested by adding
exactly the seeds the hypothesis called for, and it failed. Nothing of ours
exceeds GenScore on either metric at six seeds; the zero-selection all-families
ensemble remains R 0.8107 / rho 0.7175.

The general lesson is the one already in this file's own history: a metric read
off three seeds is a draw from a distribution, and a favourable draw is not a
result until more members confirm it.

# Audit: the aggregation was mixed across rows of the same table

Every headline this session was recomputed from the raw dumps in one place. The
basis and the core numbers reproduce exactly:

| | recomputed | quoted |
|---|---|---|
| dropped targets | 21 (8 leaked ∪ 19 reproduced) | 21 |
| clean targets | 79 | 79 |
| best arm `vina_score` | **−1.308** | −1.308 |
| best arm `vina_min` | **−4.759** | −4.759 |
| unique SMILES | 0.691 | 0.691 |
| reference `vina_score` | **−6.871** | −6.871 |

Two things did not, and both are aggregation, not data.

## The same row used three different aggregations

| quantity | what was quoted | the aggregation it came from |
|---|---|---|
| `vina_score` −1.308 | median of per-target medians | |
| strain 870 | **mean** of per-target medians (the pooled median is **531**) | |
| PoseBusters 0.515 | mean of per-target *means* (pooled is 0.517) | |

All three are defensible; using three in one table is not. And none of them is
what the bench's own `per_model.csv` reports, which is the **mean** of per-target
medians: **−0.638** on the clean 79, **−1.218** on all 100.

The headline moves **0.67 kcal** on the choice alone:

| aggregation | `vina_score` |
|---|---|
| median of per-target medians | −1.308 |
| **mean of per-target medians** (`per_model.csv`) | **−0.638** |
| pooled median over molecules | −1.409 |

## The ladder mixed marginal and paired differences

`vina_dock`'s gap was computed **paired** (+0.447) while `score` and `min` were
computed as differences of **marginal medians** (+5.563, +2.151) — so the three
rows of one decomposition were not the same kind of number. Redone paired
throughout, which is the only version that differences the same targets on both
sides:

| | arm | reference | **paired gap** | p |
|---|---|---|---|---|
| `vina_score` | −1.308 | −6.871 | **+5.108** | 1.3e-13 |
| `vina_min` | −4.759 | −6.912 | **+1.717** | 2.1e-11 |
| `vina_dock` | −7.287 | −7.859 | **+0.447** | 6.9e-04 |

| | kcal | share |
|---|---|---|
| removed by local optimisation | +3.391 | **66%** |
| further removed by redocking | +1.271 | **25%** |
| **left over — molecule quality** | **+0.447** | **9%** |

against the 5.563 / 62% / 31% / 8% quoted before. **The conclusion is unchanged
— placement is 91% and the molecule is 9%** — but these are the numbers to use.

## What to carry forward

* **Every arm-to-arm comparison this session was already paired**, so none of
  them is affected. Only the gap-to-reference shifts, 5.563 → **5.108**.
* Quote the paired difference and say so; a difference of marginal medians is a
  different quantity and moves by half a kcal here.
* `per_model.csv` is a **mean** of per-target medians. Anything compared against
  a number from that file has to use the same aggregation.

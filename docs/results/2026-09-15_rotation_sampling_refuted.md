# Rotation sampling: the lever does not work by its own rule (S1, S2 both False)

2026-09-15 02:30 JST. Pre-registered in
`docs/notes/2026-09-15_prereg_rotation_sampling.md` before any head was trained;
applied by `experiments/affinity_loop_20260914/rotation_sampling_verdict.py`,
which was written while the heads were still queued. Raw output in
`rotation_sampling_verdict_output.txt`.

Before the verdict: all 8 heads verified (checkpoints, clean log end, `qacct`
exit 0) and all 16 inference jobs verified (`qacct` exit 0, no log errors,
285/285 non-NaN). The mechanism check that the flag actually took effect passed
first: **every head, treatment and control, ran 254 steps per epoch** (8,150
complexes / 32), so the treatment trained on complex-sized epochs drawing one of
8 orientations, not on 8x the documents.

## Verdict: DOES NOT WORK

| f32 family ensemble | R | between | within | rho |
|---|---:|---:|---:|---:|
| treatment `e250_rot8_sampled` | 0.8040 | 0.3662 | 0.4378 | 0.7158 |
| control `e250_rot1_ctl` | 0.7851 | 0.3368 | 0.4483 | 0.6912 |

* family dR **+0.0189**, family drho **+0.0246** -- both clear the +0.010 bars.
* paired dR: s7 -0.0034, s11 +0.0020, s12 +0.0160, s13 -0.0028 -> **2/4** (needs 3/4)
* paired drho: s7 -0.0754, s11 -0.0193, s12 -0.0070, s13 +0.0263 -> **1/4** (needs 3/4)
* f16 agrees: dR +0.0143 (1/4), drho +0.0193 (1/4).

**S1 and S2 both fail.** The family ensembles improve; the individual seeds do
not. Mean per-seed effect is dR +0.003 and drho **-0.019** -- the ranking gain
exists only after the four seeds are pooled.

## M1 (mechanism) holds, which makes the result informative rather than null

Final train/loss, treatment vs control: s7 0.173 vs 0.0895, s11 0.0904 vs
0.0551, s12 0.256 vs 0.0841, s13 0.554 vs 0.258 -- **higher on 4/4 seeds**. The
augmentation does what it was argued to do: it makes the training set harder to
memorise (the control memorises to train/loss 0.055-0.26 against val ~0.7). So
this is not "the flag did nothing"; it is "less memorisation, no reliable
per-seed gain".

## What this most likely is, stated as a hypothesis and not a result

Rotation sampling looks like it raises **seed diversity** rather than per-seed
quality: each head sees a different frame sequence, the four heads decorrelate,
and only the pool improves. The between-target part rose +0.029, which is the
component we actually lose on -- but a 4-seed family ensemble is a noisy
statistic (extending `union_noconf` from 3 to 6 seeds moved rho by -0.026), and
the pre-registration asked for per-seed consistency precisely because family
means have fooled this project before. Recorded as a refutation; if it is ever
revisited it needs more seeds, pre-registered as a seed-diversity claim rather
than a per-seed one.

Fourteen levers are now refuted or neutral.

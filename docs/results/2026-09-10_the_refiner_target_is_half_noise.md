# The refiner's target is not deterministic — 60% of it is search noise

2026-09-10. Measured on 120 training records after asking why the refiner's
step is a fixed 0.38 of its target's magnitude when the target is supposed to be
a function of the inputs the network already sees.

## The claim that fails

`build_overlap_targets.py` justifies the whole overlap-target design this way:

> With `x1` = the crystal pose, the target is NOT determined by what the network
> sees … The bounded transform that relieves the receptor overlap **IS a
> deterministic function of exactly those inputs.**

`rigid_pocket_fit` is Powell from `n_restarts` random starting points
(`n_restarts=4`, `seed=0` as deployed). Running it twice on the same record with
only the seed changed:

| | median | mean | p90 |
|---|---|---|---|
| target magnitude (move from x0) | 2.021 Å | | |
| **seed 0 vs seed 1, 4 restarts** | **1.207 Å** | 1.324 | 2.925 |
| 4 restarts vs 16, same seed | 0.418 Å | 0.831 | 2.112 |

**Noise is 59.8% of the target's own magnitude, and 0.0% of 120 records give
the same answer to within 0.01 Å.** The target is the output of a stochastic
search, not a deterministic function.

## This explains the 0.38 exactly

A least-squares regressor shrinks its output toward the conditional mean by the
fraction of the target's variance it can explain. With 60% of the target being
seed noise no network can see, the explainable fraction is about 0.4 — and the
deployed refiner's step is **0.38 of its target's magnitude** (0.754 Å against
1.96), measured independently in
`2026-09-09_the_step_follows_the_scale_of_the_target.md`.

The step is not short because the model hedges across modes, nor because its
receptive field is too small (it is not: the ligand graph is one component of
diameter ≤ 5 against 5 layers, and each atom sees the nearest 32 pocket atoms
within 8 Å). **It is short because most of what it is asked to predict is
unpredictable.**

## What this puts upstream of everything

Eight refiner settings were run and seven lost:

| what was changed | outcome |
|---|---|
| target = crystal pose | loses, p=8.9e-04 |
| target = LJ well | loses, p=5.7e-09 |
| target with rotation removed | loses, p=0.035 |
| head = rotation / torsion | does not converge |
| step magnitude (α) | flat, then collapses |
| step count (rounds) | null |
| training-input rotation cap | loses, p=3.7e-05 |
| schedule t≡0 | the one that works |

Every one of them aimed at a target that is 60% noise. That does not make any
of those results wrong — they are all measured against the same control under
the same conditions — but it does mean none of them tested what it was built to
test at full strength.

## What it licenses, and what it does not

**Does not**: inflating the target, which is arithmetically the same as scaling
the output and was measured flat-to-collapsing; nor a force field at inference,
which the August policy excludes and the 2026-09-09 decision reaffirmed.

**Does**: converging the search. 4 → 16 restarts already cuts the disagreement
from 1.207 Å to 0.418. This changes only how accurately an existing geometric
quantity is solved — the same objective, the same bounds, no new weight, no new
module, nothing at inference. It costs one corpus rebuild.

How many restarts are needed, and how much noise is irreducible, is being
measured against a 256-restart reference before any arm is registered.

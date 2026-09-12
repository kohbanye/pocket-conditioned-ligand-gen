# Which target the gap is in, and the model does adapt after all

Every analysis this session asked *which term* (repulsion, 71%) or *which atom*
(interior, degree 2, 59.5%). Nobody asked **which target**.

## The gap is skewed, not uniform

Best arm (`dfs + MLM + clash + prune`) median minus the crystal reference's own
`vina_score`, per target, clean 79:

| quantile | gap |
|---|---|
| 10% | +2.111 |
| 50% | **+5.108** |
| 90% | +10.120 |
| 100% | **+34.469** |

mean +6.034, sd 5.237. The worst 5 targets carry 22.3% of the summed gap, the
worst 20 carry 50.6%.

| target | gap | arm median | arm best | reference |
|---|---|---|---|---|
| IDHP_HUMAN_40_452_0 | +34.47 | +23.13 | −1.39 | −11.34 |
| P2Y12_HUMAN_1_342_0 | +22.22 | +11.79 | −1.75 | −10.43 |
| TNKS1_HUMAN_1099_1319_0 | +18.12 | +3.10 | **−6.63** | −15.03 |
| AKT1_HUMAN_1_137_0 | +11.96 | −2.19 | **−9.85** | −14.15 |
| ... | | | | |
| CDK6_HUMAN_1_312_0__4aua_A_rec | **−4.28** | −3.11 | −5.57 | **+1.17** |
| CHIB_SERMA_1_499_0__1h0i_A_rec | **−4.59** | −1.03 | −3.96 | **+3.56** |

The reference's own score spans **+3.56 to −15.03**. The two targets the arm
already beats are the two where the crystal ligand scores *positive* in its own
receptor. So a large part of "the gap" is how good the crystal ligand happens to
be, which is not a property of the model at all.

## What predicts a target's gap

Spearman over the 79:

| predictor | rho | p |
|---|---|---|
| **clash count** | **+0.675** | 9.3e-12 |
| **reference's own score** | **−0.618** | 1.3e-09 |
| reference ligand heavy atoms | +0.493 | 3.8e-06 |
| pocket atoms | +0.382 | 5.2e-04 |
| PoseBusters rate | −0.355 | 0.0013 |
| strain | +0.312 | 0.0052 |
| blind atoms (the CA-rule cut) | +0.221 | 0.051 |

Clashes first, and the reference's own quality second.

## The inference that was wrong, and the measurement that fixed it

The arm's raw `vina_score` is **uncorrelated** with the reference's score
(rho +0.118, p=0.3) and with the reference ligand's size (rho −0.038, p=0.74).
Read alone that says the model produces target-independent quality and does not
use its conditioning — which would contradict the pocket-swap control.

It is clash noise. The same correlation on `vina_min`, which is the same
molecule after local optimisation:

| arm quantity | vs the reference's score | rho | p |
|---|---|---|---|
| `vina_score` | | +0.118 | 0.3 |
| **`vina_min`** | | **+0.483** | **6.5e-06** |

**The model does adapt.** And it adapts in size too: heavy atoms against the
reference's score gives rho **−0.605** — where the crystal ligand binds better
(more negative), the arm builds *larger* molecules. Right direction, strong
effect.

## The one number this session should be read by

| | arm | reference | gap |
|---|---|---|---|
| `vina_score` | −1.31 | −6.87 | **5.56** |
| `vina_min` | −4.76 | −6.91 | **2.15** |
| per-target sd | 4.00 / 1.90 | 2.97 | |

**61% of the reported gap disappears under local optimisation**, and the arm's
per-target spread in the raw score (sd 4.00) is *larger* than the reference's own
spread (2.97) while its spread in `vina_min` (1.90) is smaller. Most of what the
reported metric measures about this arm is clash noise, not chemistry and not
target difficulty.

The molecules are much better than their poses. Closing a pose gap needs an
optimiser with an objective in translation-rotation-torsion space, which is the
force field, excluded by standing decision — the same wall reached from three
other directions today.

## What the worst targets share: size, not placement

| predictor of the per-target gap | rho | p |
|---|---|---|
| clash count | +0.675 | 9.3e-12 |
| **reference ligand's radius of gyration** | **+0.462** | 1.8e-05 |
| centroid offset from the reference ligand | +0.236 | 0.036 |

The generated molecules sit in the right place — centroid offset is 1.28 A at the
median, and it barely predicts anything. **Every one of the eight best targets
has a median clash count of 0.0**; the worst ten run from 0 to 12.5.

So the chain is: a target whose crystal ligand is large gets a large molecule
from the arm (rho −0.605, the right response), a large molecule has more atoms to
place, and the arm cannot place them.

## And the clash rate per atom rises with size, which the reference's does not

Fraction of heavy atoms clashing with the receptor, by molecule size:

| heavy atoms | reference | generated |
|---|---|---|
| 0–15 | 1.43% | 1.71% |
| 15–20 | 0.42% | 2.17% |
| **20–25** | 0.00% | **3.93%** |
| 25–30 | 0.57% | 3.79% |
| 30–40 | 0.64% | **4.13%** |
| 40+ | 0.00% | 4.12% |

| | rho(size, clash rate) | p | overall |
|---|---|---|---|
| reference | +0.095 | 0.4 | 0.60% |
| **generated** | **+0.181** | **4.3e-13** | 3.14% |

The generated rate more than doubles by 20 atoms and then plateaus near 4%. The
reference's shows no trend — though on only 79 molecules, so that half is
under-powered and the honest claim is the generated trend alone.

**This is not extrapolation.** The training corpus is not short of large ligands:
over 1,269,592 ligand blocks in `data/lm_tokens_dfs`, the median is **27 heavy
atoms**, 78.9% have ≥20 and 36.6% have ≥30. The model generates a median of 22.4,
*smaller* than its own training distribution — and close to the canonical 100's
reference median of ~23, which is another sign it takes its size from the pocket
rather than from the corpus.

So size-dependent clashing is a failure of **conformational fit**: the molecule's
internal geometry does not match the pocket's shape, and a rigid placement has six
degrees of freedom with which to satisfy 30 atoms. That is the same conclusion the
degree analysis reached from the other side — interior atoms carry 59.5% of the
repulsion — and it needs an optimiser with an objective in conformation space.

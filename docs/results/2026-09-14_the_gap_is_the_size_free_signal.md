# The gap is in the size-free signal, not in size calibration

2026-09-14. With GIGN and OnionNet-2 measured here on the same 285 complexes,
the deficit can be split into the part ligand size explains and the part it does
not. The split reverses the standing framing.

## The decomposition

Residualise both the prediction and the label on heavy-atom count, then
correlate what is left:

| | full R | **size-free R** | corr(heavy atoms, prediction) |
|---|---:|---:|---:|
| ours, all-families ens(45) | 0.8107 | **0.7368** | +0.626 |
| OnionNet-2 | 0.8513 | 0.8106 | +0.446 |
| GIGN (z-sum of 3) | 0.8550 | **0.8339** | +0.366 |
| true pK | — | — | +0.5005 |

Heavy-atom count alone predicts pK at R = 0.5005.

Two things follow, and the second is the one that matters.

**We are the only one that over-uses size.** Our predictions correlate with
ligand size at +0.626 against a true +0.500. Both models that beat us sit
*below* the truth: GIGN +0.366, OnionNet-2 +0.446.

**But removing size widens the gap rather than closing it.** Against GIGN the
full-R deficit is −0.044; on the size-free component it is **−0.097**. Our
non-size signal is 0.7368 where GIGN's is 0.8339.

## Residual bias by ligand size says the same thing backwards

Mean signed error (predicted − measured), by heavy-atom count:

| | <=20 heavy (n=109) | 21-35 (n=135) | >35 (n=41) |
|---|---:|---:|---:|
| ours | −0.022 | +0.009 | +0.028 |
| OnionNet-2 | +0.310 | −0.085 | −0.544 |
| GIGN | +0.439 | −0.158 | −0.647 |

Our residuals are **flat in size**; the two winners' are strongly sloped. They
under-use size and pay for it with size-correlated error — and still score
higher, because what they put in its place is worth more than what we put in
its place.

By affinity the three are alike: everyone over-predicts weak binders and
under-predicts strong ones (ours +0.879 / −1.374 at the extremes, GIGN +0.727 /
−0.990). Range compression is shared, not ours.

## What this settles

Interventions aimed at our size dependence -- efficiency training, projecting
size out of the representation, recalibration -- are now firmly ruled out, and
for a sharper reason than before. It was already known that removing size hurt
(iterative projection took pK prediction 0.468 → 0.358) and that optimal
reweighting was worth +0.0005. This adds why: **we are already better
size-calibrated than either model that beats us.** There is nothing to fix
there. The deficit is elsewhere.

The target is now a specific quantity: **+0.097 of size-free R**, which is
signal about the complex that is not "how big is the ligand". That is what the
tokenizer and the LM representation are supposed to carry, which is where the
remaining effort belongs.

## Error decomposition, for scale

| | RMSE (pK) | MAE |
|---|---:|---:|
| ours | 1.2706 | 1.0207 |
| OnionNet-2 | 1.1388 | 0.8989 |
| GIGN | 1.1256 | 0.8481 |

corr(our error, GIGN error) = +0.599, so of our error variance 1.620, about
**36% is shared with GIGN and 64% is ours alone**. Per cluster we beat GIGN's
rho on 18 of 57, tie on 15 and lose on 24 — the ranking deficit is distributed,
not carried by a few targets.

The twelve complexes where we lose most to GIGN carry 26.4 pK of the 118.7 total
excess error, i.e. the loss is broad rather than concentrated. Ten of the twelve
are weak binders we over-predict; the two exceptions (3f3e, 3f3d, both 9 heavy
atoms, pK 7.70 and 7.16) are small strong binders we under-predict by ~3 pK.

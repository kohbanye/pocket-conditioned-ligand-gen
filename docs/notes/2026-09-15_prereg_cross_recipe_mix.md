# Pre-registration: mixing two recipes beats more seeds of one -- on fresh seeds

2026-09-15 ~16:15 JST. Written BEFORE any of the 16 fresh heads (seeds 18-25 of
each recipe) is trained and before any CASF number for them exists.

## Origin -- disclosed

An exploratory pass ON CASF, run after the rotation-ensemble verdict, found:

| f32 family z-sum | n | R | between | within | rho |
|---|---|---|---|---|---|
| `e250_unionnc_corr` | 8 | 0.8118 | 0.3716 | 0.4402 | 0.6947 |
| `e250_unionr8_ens` | 8 | 0.8093 | 0.3852 | 0.4241 | 0.6930 |
| corr 4 + unionr8 4 | 8 | 0.8124 | 0.3783 | 0.4342 | 0.6912 |
| **corr 8 + unionr8 8** | 16 | **0.8165** | 0.3811 | 0.4354 | **0.7246** |

Those numbers chose nothing on CASF (the mix rule is "all seeds of both
families"), but they were seen, so they are exploratory. This round re-runs the
same rule on **seeds that do not exist yet**.

Why it is plausible rather than a fluke of size: within one family, 4 -> 8 seeds
gains +0.0008 (`docs/results/2026-09-15_ensemble_size_saturated.md`), so the
+0.0047 at n=16 is not what same-family seeds buy. The two recipes differ in
what they get wrong -- rotation training raises between-target R to 0.3852 and
lowers within to 0.4241, the plain recipe is the mirror image -- and their
family residuals correlate 0.945, lower than 0.984 between the two non-rotation
families.

## Arms (fixed now)

Fresh seeds **18, 19, 20, 21, 22, 23, 24, 25** for each recipe, identical to the
recipes already registered, only the seed differs:

* `e250_unionnc_corr_v2`: `data/lm_tokens_e250aff_union_noconf`, `--corr-weight 0.7`
* `e250_unionr8_ens_v2`: `data/lm_tokens_e250aff_union_r8`, `--rotations-per-complex 8`

Both: `--label-cap 13 --max-epochs 20 --early-stop-patience 4 --block-size 640`,
backbone `mlm_e250lig3` e02, ligand readout, f16 and f32 inference.

## Predictions (f32 basis, fresh seeds only -- no seed from the exploratory pass)

* **M1 (scoring).** R(mix of 8+8 fresh) - max(R(corr 8 fresh), R(unionr8 8
  fresh)) >= **+0.004**.
* **M2 (ranking).** rho(mix) >= max(rho(corr 8 fresh), rho(unionr8 8 fresh)).

**The lever works only if M1 AND M2 hold.** f16 is reported beside; because rho
moves +-0.02 per seed, a pass at f32 with a fail at f16 is reported as such and
does not become a headline on its own.

Reported beside, not part of the verdict: the 32-model mix (old + fresh seeds of
both recipes), and the goal line (R > 0.8127, rho > 0.7281, R > 0.83).

## What this cannot become

No headline from this note. If M1 and M2 hold, a headline needs its own
pre-registration, and it must state the model is a 16-model two-recipe ensemble
and compare against GenScore's own 3-checkpoint ensemble, not only its single
released checkpoint.

## Cost

16 head jobs (gpu_1, 1:00) ~= 3.2 pt + 32 inference jobs (gpu_1, 0:30) ~= 3.2 pt
~= **6.4 pt**, taking the ledger to about 127 of the 200 pt cap.

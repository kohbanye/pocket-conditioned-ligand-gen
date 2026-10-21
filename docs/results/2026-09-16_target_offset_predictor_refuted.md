# Predicting the per-target offset from covariates: refuted by a nested test

2026-09-16 ~02:00 JST. Exploratory (reads CASF labels), zero compute cost, run
on existing dumps. This closes the route that the ceiling calculation had
appeared to open.

## Why the route looked open

The ceiling calculation on the 36-model mix (f32, R 0.8229, target-bias sd
0.629 pK) showed R 0.830 is reached by removing only **k = 0.072** of the target
bias, and that the bias is concentrated: 13 targets are *coherent* offsets
(|bias| > 2 SE, within-target sd 0.3-0.65 against biases of 0.8-1.75) carrying
**56%** of the squared bias. Removing only those 13 gives R 0.8512; only the top
5 gives R 0.8442. rho is unchanged by construction, so any such correction would
satisfy all three conditions at once.

Since subtracting beta * ghat at the optimal beta removes rho^2 of the bias
variance, the required predictor correlation was only **rho ~ 0.27**.

## The measurement

Covariates, all label-free or training-corpus-derived (CASF-label columns
`logka`, `pK`, `z_label`, `res_ours`, `abs_ours`, `err_ours`, `ae_ours`,
`学習-CASF_ours` explicitly blacklisted):
14 ligand descriptors (heavy, mw, logp, tpsa, rotb, hbd, hba, fsp3, rings,
arom_rings, charge, n_P, n_S, n_halogen) + `n_ours` (training coverage) +
`trainpK_ours` (training mean pK) + `pred_mean` (our own prediction) +
`disagree` (sd across the 36 heads after z-scoring).

| procedure | LOO rho | corrected R |
|---|---|---|
| lambda picked by reading the LOO column (lambda=0.1) | +0.442 | 0.8323 |
| **fully nested: lambda AND beta chosen inside each fold** | **+0.088** | **0.8215** |

**Permutation null** (200 shuffles of the target offsets, same fully nested
procedure): mean R 0.8202, sd 0.0046, 95th pct 0.8245, max 0.8275.
Observed 0.8215 → **p = 0.527**. Gain over uncorrected: **-0.0013**.

**0 of 200 null draws reached 0.83.**

## What went wrong the first time, and it is a repeat offence

rho = 0.442 was produced by selecting lambda from 8 candidates *on the LOO
statistic itself*, with 57 targets and 18 covariates. That is selection on the
test statistic — the same family of error as `feedback_measurement_traps.md`.
The honest procedure nests the selection, and the signal vanishes.

Corroborating detail: at lambda=0.1 the two largest ridge coefficients were
`mw +1.634` against `heavy -1.530` — a near-collinear pair with opposite signs,
the standard signature of a ridge riding a difference it cannot support at this
sample size.

## Sub-results, both negative

* **Ensemble disagreement is not a proxy for error.** corr(disagree, |bias|)
  = +0.234 per target, +0.160 per complex. The 36 heads do not disagree where
  they are wrong. This covariate was never in the earlier sweep; it is now.
* **The offset is not a recalibration artefact.** `pred_mean` alone gives LOO
  rho **-0.078**. Our predictions are not simply shrunk as a function of their
  own magnitude, so no fair recalibration (fittable on training/val data, no
  CASF labels) exists.

## A fairness point worth stating

Even the optimistic 0.8323 was fitted on the *other 56 CASF targets' labels*.
That is a feasibility probe, never a headline number: a CASF result may not be
calibrated on CASF labels. A fair version would need out-of-fold offsets from
the training corpus. The nested result removes the reason to build one.

Blocking obstacle if anyone retries: the teacher's training-set predictions are
in-sample (pearson 0.9885 vs labels), so training-target offsets have sd
**0.1404 pK** against CASF's 0.629, and residual sd 0.2919 against 1.233. Honest
training offsets require K-fold out-of-fold heads.

## Identity of the 13 coherent targets — no shared family

HIV-1 integrase (+1.75), transporter (-1.26), cGMP phosphodiesterase (-1.23),
anthranilate phosphoribosyltransferase (-1.11), trypsin beta (-1.03),
beta-lactamase (+0.92), urokinase plasminogen activator (-0.78), PTP1B (+0.77),
pantothenate synthetase (+0.75), RNA-directed RNA polymerase (-0.70), p53
(+0.64), androgen receptor (-0.64), tankyrase-2 (-0.45).

Only trypsin/uPA share chemistry (both serine proteases, both under-predicted).
Too thin to build on.

## One correction to the record

Our per-target bias sd is **0.629** vs GenScore's **0.715** — we are *less*
target-biased than GenScore. Our deficit is within-target: sd **0.945** vs
GenScore **0.855**. Earlier framing of our target bias as pathological was
wrong; it is ordinary, and R > 0.83 is simply a high absolute bar.

## Cost

Zero. All analysis ran on existing dumps. Ledger unchanged at **179.18/200 pt**
(20.82 remaining).

---

# Addendum: within-target compression, also refuted (lever #30)

Same night, ~02:30 JST. Zero compute.

The per-complex breakdown of the worst-scatter targets appeared to show our
predictions failing to span the label range within a target:

| target | label range | ours | GenScore |
|---|---|---|---|
| transporter (c17) | 3.51 | **0.92** | 1.86 |
| thermolysin (c27) | 7.40 | **2.49** | 4.09 |
| thrombin (c34) | 7.64 | 3.39 | 5.43 |

A within-target rescale `pred' = target_mean + c*(pred - target_mean)` preserves
ordering exactly (rho invariant) and would have been fair to fit on training
data. It does not work:

* **Within-target slope (label ~ centred pred): ours 1.090, GenScore 1.115.**
  GenScore is *more* compressed than we are. We are essentially calibrated.
* Sweeping c: best is c=1.2 → R 0.8260 (**+0.0031**), and c=1.2 was chosen by
  reading the R column; the analytic optimum is c=1.090. GenScore gains
  comparably (0.8127 → 0.8176). rho constant at 0.7316 throughout, as designed.

**The error was circular selection.** Those three targets were chosen *because*
they had the worst within-target scatter, so of course they look compressed.
Globally the effect is absent.

## The defect hypothesis is also refuted

HIV-1 protease (within-sd 1.521 vs GenScore 0.475) suggested an implementation
bug, in the style of the `parse_sdf` counts-line defect. It is not one:

* complexes with |resid| > 2.5: **ours 12, GenScore 14** — we have *fewer*
  extreme outliers.
* heavy atoms in that group 26.9 vs 24.5 overall; disagreement 0.304 vs 0.263.
* no complete separation, no single broken parse, outliers spread over 10
  different proteins.

## Standing position after 30 levers

W1 holds: the 36-model mix beats GenScore's released checkpoint on **both**
metrics (R 0.8229 vs 0.8127, rho 0.7316 vs 0.7281), surviving the twin screen
(0.8261/0.7643 vs 0.8184/0.7589). W3 (R > 0.83) is not reached, and the
structural measurements now enumerate why rather than guess.

Open caveat for any headline: `vq_e250_lig3`, the tokenizer under every e250
arm, has unresolved CASF membership — the same-lineage V1 checkpoint had 169 of
285 CASF complexes on its train side, and `fairness.md` records that the
exclusion flag is opt-in and not stored in the checkpoint. This is *unsupervised*
exposure (no affinity labels; CASF labels were held out of head training), but
it should be stated, and `vq_ctrl_p3` exists as a leak-free tokenizer for a
robustness check.

Ledger unchanged: **179.18/200 pt**, 20.82 remaining. Nothing was submitted
tonight after the ceiling calculation, because every remaining candidate is
predicted to fail by these same measurements.

---

# Addendum 2: leak check at the MLM layer via ligand pseudo-log-likelihood

2026-09-16 ~03:00 JST. Zero compute — `pll` is already dumped in every result
CSV (`pdbid,logka,cluster,pll,head`), computed at `k==0` in
`inference._score_complex`.

Motivation: `vq_e250_lig3`, the tokenizer under every e250 arm, has unresolved
CASF membership (same-lineage V1 had 169/285 CASF complexes on its train side;
the exclusion flag is opt-in and not stored in the checkpoint). Swapping to the
leak-free `vq_ctrl_p3` would change the token vocabulary and require retraining
the MLM and all 36 heads — far beyond the remaining 20.82 pt. So the question
was tested where it actually matters: **if the model memorised these complexes
in a way that helps affinity, familiarity should predict accuracy.**

`pll` (mean over all 36 arms; cross-arm correlation 1.000, i.e. it is a property
of the MLM, not the head): mean -3.115, sd 1.475, range -7.835..-0.933.

| relationship | value |
|---|---|
| corr(pll, \|residual\|) per complex | **-0.056** |
| corr(pll, signed residual) | -0.066 |
| corr(pll, logka) | +0.143 |
| corr(pll, target bias) per target | -0.338 |
| corr(pll, \|target bias\|) per target | -0.244 |

| pll quartile | mean \|residual\| |
|---|---|
| top 25% (most familiar, n=72) | 1.045 |
| bottom 25% (least familiar, n=72) | 1.094 |

**Familiarity buys ~4% of accuracy — noise.** There is no evidence that
memorisation transferred into the affinity prediction.

Stated with its limits: `pll` probes *ligand-token* likelihood under the MLM,
not pocket memorisation inside the VQ, so this weakens the concern at the layer
that feeds the head without proving absence upstream. The target-level -0.338 is
~2.5 SE on 57 points and nothing is built on it. The honest disclosure for a
headline remains: the VQ tokenizer's CASF membership is unverifiable from the
checkpoint, it is unsupervised exposure with no affinity labels, CASF labels
were held out of head training, and familiarity does not predict accuracy.

---

# Addendum 3: label noise is not the cause of weak single models (lever #31)

2026-09-16 ~03:30 JST. Zero compute, refuted before any spend.

**Reframing that motivated it.** Every lever tested tonight and before operated
on the *mix*. But the deficit is per-model: our mean single seed is R **0.7920**
against GenScore's single released checkpoint at **0.8127**. Ensembling gains are
comparable (ours +0.031 from mixing 36; theirs +0.026 going 1→3 checkpoints), so
the gap is in the individual model, which no mix-level lever can reach.

Supporting evidence that the models generalise poorly rather than underfit:
teacher heads reach pearson **0.9885** against training labels while scoring
R≈0.79 on CASF; selected checkpoints sit at val loss ≈0.45 (RMSE ≈0.67 pK)
against a CASF residual sd of **1.233**.

**Hypothesis:** the PDBbind builder applies no pK band (real training range
0.40–15.00 vs CASF ≈2–12), so implausible labels could be injecting noise that
degrades single-model quality invisibly to mix-level levers.

**Refuted.** Out of 15,351 training documents:

| band | outside | share |
|---|---|---|
| [2, 13] | 72 | **0.47%** |
| [2, 12] | 88 | 0.57% |
| [1, 14] | 8 | 0.05% |
| [3, 12] | 621 | 4.05% |

69 documents below pK 2 (0.40–1.30) and 3 above 13 (13.82, 13.96, 15.00).
Filtering 0.47% of a corpus cannot move per-model quality. The frequent repeated
values (pK 8.70 ×126, 8.00 ×125, 9.00 ×123, all <1% each) are ordinary rounding
in reported affinities, not corruption.

**What the distributions do show, and it is not noise:**

| split | n | mean | sd | 5% | 95% |
|---|---|---|---|---|---|
| train | 15351 | 6.31 | **1.93** | 3.15 | 9.41 |
| val | 431 | **7.09** | 2.05 | 3.68 | 10.39 |
| CASF | 285 | 6.49 | **2.17** | 2.86 | 10.04 |

CASF spans a wider affinity range than the training corpus, and the validation
split used for checkpoint selection sits ~0.8 pK above train. Both are
distribution shift, not label corruption.

---

# Addendum 4: data scale is not the binding constraint (rediscovered, already refuted)

2026-09-16 ~04:00 JST. Zero compute. **This addendum records a rediscovery, not
a new result.**

Following the per-model framing, I reasoned from the corpus builder's help text
(`'KD,KI' matches CASF exactly but halves the data`) that excluding IC50 had cost
us ~57% of PDBbind training data, and proposed it as the first lever aimed at
single-model quality. PDBbind v2020 breaks down as 19,443 total = Kd 7,249 +
Ki 5,004 + **IC50 7,190 (37%)**, so the size of the omission was real.

**It had already been tested and refuted on 2026-09-14**, pre-registered,
seed-paired and leak-audited
(`docs/results/2026-09-14_pdbbind_ic50_in_the_union_refuted.md`):

| f32 family ensemble | R | between | within | rho |
|---|---:|---:|---:|---:|
| `e250_unionic50` (24,895 train docs) | 0.8029 | **0.3611** | 0.4417 | 0.6737 |
| `e250_unionkdki` (14,372 train docs) | 0.8083 | **0.3719** | 0.4364 | 0.6965 |

dR **-0.0055**, drho **-0.0228**; both pre-registered conditions failed.
`variants.py:579` already recorded that "the whole e250 generation passed
KD,KI".

**The substantive point stands independently of my error: +10,523 complexes
(+73% of the corpus) made the model worse, and the between-target component —
the weak half — moved the wrong way (0.3719 -> 0.3611).** Data scale is not the
binding constraint. That is independent support for the representation being the
ceiling.

Also refuted the same hour, for free: **label noise**. Only 72 of 15,351
training documents (0.47%) fall outside pK [2,13]. What the distributions do
show is shift, not corruption: train sd 1.93 vs CASF 2.17, and the val split
used for checkpoint selection sits ~0.8 pK above train.

## Process failure worth recording

I searched `pocket-ligand-rescore/*/run.json` for corpus names, got 0 hits, and
read that as "never trained". That is a false negative: CLAUDE.md names
`variants.py` as the single source of truth for which arm points at which
weights, and the arms (`e250_unionic50`, control `e250_unionkdki`) were
registered there all along. Grep `docs/results/` and `variants.py` **before**
reasoning about a lever, not after.

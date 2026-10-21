# 83% of our per-target bias is shared across every recipe — and no covariate explains it

2026-09-16 ~00:15 JST. Exploratory (reads CASF labels), run on existing dumps at
no compute cost while two pre-registered rounds were in flight. This is a
measurement of *why* the loop's levers keep failing, not a lever itself.

## The measurement

For each recipe family, the 8-seed ensemble's residual after least-squares
calibration onto pK, averaged per CASF target → a 57-dimensional "target bias"
vector. Then the correlation between those vectors.

| pair | correlation of target bias |
|---|---|
| our families with each other (6 families) | **+0.82** (range 0.56–0.99) |
| our families vs GenScore / GIGN / OnionNet-2 | +0.46 |
| `heavy_v3` vs `corr_v3` (different pocket definition) | 0.88 |
| `frozen` vs `corr_v3` (no fine-tuning at all) | 0.78 |
| `frozen` vs GIGN | 0.06 |

**83% of our target-bias variance is common to all families.** The shared
component has sd **0.660 pK**; what is family-specific has sd 0.276 pK.

## What follows immediately

* **Ensembling can only touch the 0.276.** That is what an ensemble averages
  away. The 0.660 is identical in every member, so it survives untouched. This
  retro-explains the whole loop: 24 refuted levers, almost all of which were
  recipe-level and therefore only ever moved the 17% part.
* **The bias is not created by fine-tuning.** The frozen-encoder family, which
  trains 0.59M parameters instead of 105M, still misses the same targets in the
  same direction (0.78 with `corr_v3`) while sharing almost nothing with GIGN
  (0.06). It comes from how the complex is represented, not from how the head is
  trained on it.
* **It is largely ours.** At +0.46 against the baselines, roughly half the
  direction is shared with everyone (genuinely hard targets), but the larger
  part is specific to this model family. On the worst targets the gap is stark:
  cluster 52 ours −2.03 vs baselines −0.14; cluster 17 +1.21 vs +0.52.

## What the bias is *not*

| candidate explanation | correlation with the shared bias |
|---|---|
| target's mean pK (shrinkage toward the corpus mean) | +0.360 (slope +0.169) |
| rotatable bonds | −0.343 |
| molecular weight / heavy atoms | −0.282 / −0.264 |
| TPSA | −0.238 |
| logP | +0.112 |
| pocket residue count | +0.105 |
| same-UniProt training complexes | −0.172 |
| training-set mean pK for the target | +0.082 |
| **all of them together (multiple regression)** | **R² = 0.306** |

**69% of the shared bias is unexplained by every property we can compute.** It
is not a size effect, not a pocket-size effect, and not a coverage effect —
targets with plenty of training data are missed about as badly as targets with
none. Shrinkage toward the corpus mean accounts for roughly a tenth of it.

## Consequence for the goal

R > 0.83 needs about +0.007 on top of the 36-model mix's 0.8229. The routes are
now enumerated rather than guessed:

1. **More ensembling** — bounded by the 0.276 family-specific component, already
   saturated (4→8 seeds bought +0.0008).
2. **More recipe levers** — same 17% ceiling; 24 refutations are consistent with
   it.
3. **A covariate to feed the model** — none exists at R² 0.31 for everything
   measurable combined.
4. **A different representation of the complex** (tokenizer or backbone) — the
   only route the measurement leaves open, and far beyond the 26 pt remaining
   under the 200 pt cap.

## Route 4 tested, and it closes too

The measurement above left exactly one route open: a different representation of
the complex. The project has one — the **first-generation arm**, a different VQ
tokenizer *and* a different MLM (`xzkjxu9q` + `wxlhgqx3`), trained on the same
task.

Its target bias correlates with our e250 shared bias at **+0.8519** — the same
as e250 families correlate with *each other* (+0.82), and far above the
baselines (+0.46). Mixing it in moves the 24-model ensemble from R 0.8244 to
**0.8247** (+0.0003, noise).

So a full generation change -- new codebook, new pretrained encoder -- does not
decorrelate the bias. The 0.660 pK offset is not a property of one codebook or
one MLM. It is a property of the approach: all-atom VQ tokens of pocket and
ligand, read by an encoder-only MLM, regressed to pK.

**Consequence, stated plainly.** R > 0.83 is not reachable by any lever
available in this architecture family, not merely by any lever affordable within
the remaining budget. Ensembling is bounded by the 17% family-specific
component; recipe levers move only that same 17% (24 refutations); no measurable
covariate explains the bias (R² 0.31 for everything combined); and the one
available change of representation behaves like no change at all. What the
competitive methods have and this one does not is explicit interaction
featurisation, which this project deliberately excludes.

Two pre-registered rounds were in flight when this was written (adaptation
budget: frozen vs frozen+LoRA vs full fine-tune; and distillation from the
8-seed ensemble). Their verdicts are written automatically by collector jobs to
`logs/verdict_adapt.log` and `logs/verdict_kd.log`.

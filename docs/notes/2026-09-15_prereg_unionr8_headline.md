# Pre-registration: the headline comparison for the rotation-ensemble arm

2026-09-15 ~15:50 JST (cluster clock). Written AFTER the 24 jobs of
`e250_unionr8_ens` were submitted and BEFORE any head finished training or any
CASF dump for the arm existed. The lever verdict is pre-registered separately
(`docs/notes/2026-09-15_prereg_rotation_ensemble.md`); this note fixes what
headline is reported and how, whatever that verdict says.

## Why a separate note

The rotation-ensemble pre-registration says a headline needs its own
registration before the ensemble is scored, and that it must state the number
is an 8-seed ensemble. The ensemble-size result (4 -> 8 seeds gains +0.0008)
and the kNN refutation both landed since; neither changes this design.

## What is compared

**Ours:** `e250_unionr8_ens`, all 8 seeds (7, 11, 12, 13, 14, 15, 16, 17), family
z-sum, **f32 is the basis**, f16 reported beside. No seed selection, no family
pooling with other arms, no blend.

**GenScore, two references:**

1. *Single released checkpoint* `GatedGCN_ft_1.0_1.pth`, 285/285 (the bench's
   existing dump with 4k18/4mme recovered): **R 0.8127, rho 0.7281**.
2. *Released-checkpoint ensemble* `GatedGCN_ft_1.0_{1,2,3}.pth`, z-sum, scored
   with the same CLI and the same two-complex recovery. This is the fair
   counterpart to an 8-seed ensemble: GenScore ships three seeds of this
   configuration, so ensemble-vs-ensemble is available. If 2 and 3 cannot be
   scored 285/285 before our dumps land, the headline reports reference 1 only
   and says the ensemble-vs-ensemble comparison is missing.

## Predictions / claims (f32 basis)

* **H1.** R(ours) > R(GenScore) AND rho(ours) > rho(GenScore), against
  reference 1; and separately against reference 2.
* **H2.** R(ours) > 0.83.

Reported beside each, not changing the verdict:

* the same numbers on the twin-clean subset (the union_noconf audit's twin set,
  as used by `headline_concat_prereg.py`; the arm trains on exactly
  union_noconf's complexes, so that audit applies unchanged);
* the mean single-seed R / rho of our 8 seeds against reference 1 (single
  model vs single model);
* the lever verdict (E1/E2) from the rotation-ensemble note.

## Disclosure fixed now

* An H1 pass against reference 1 only is "an 8-seed ensemble beats one released
  GenScore checkpoint", and must be written that way.
* Ours uses no contact pairs, interaction edges, distance bins, or contact
  counts; GenScore's score is built from learned protein-ligand distance
  distributions.
* The recipe this arm extends was chosen after seeing CASF numbers from earlier
  rounds (union corpus, label cap, head recipe). The headline is therefore
  confirmatory for this arm only, not for the design process.

## Cost

GenScore checkpoints 2 and 3: two gpu_1 jobs (~0.5 h each), ~0.2 pt.

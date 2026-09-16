# Pre-registration: the three-recipe mix, fresh seeds, judged once

2026-09-15 ~19:35 JST. Written BEFORE any of the 24 fresh heads exists and
before any CASF number for them exists.

## Why this composition, and why it is not selection

Two pre-registrations passed on fresh seeds:

* `docs/results/2026-09-15_cross_recipe_mix_confirmed.md` -- mixing the plain
  (correlation-loss) recipe with the rotation recipe: M1 +0.0080, M2 +0.0140.
* `docs/results/2026-09-15_heavy_third_recipe_confirmed.md` -- adding the
  heavy-atom-pocket recipe at fixed ensemble size: T1 +0.0045, T2 +0.0018.

The mix below is exactly those three recipes in equal numbers. It is what the
two passed verdicts imply, not a configuration picked by scanning CASF. Two
other axes were pre-registered and refuted (all-atom readout, longer training),
and a fifth (balanced readout) was killed on existing dumps before spending.

**Disclosed:** several 24-model mixes have already been seen clearing GenScore's
single checkpoint on one frame setting or another, as by-products of those
verdict runs. That is why this round uses **entirely fresh seeds** and fixes
the basis in advance.

## Arms (fixed now)

Eight fresh seeds per recipe, all on the recipes exactly as already registered:

| family | corpus | recipe flag | seeds |
|---|---|---|---|
| `e250_unionnc_corr_v3` | `union_noconf` | `--corr-weight 0.7` | 58-65 |
| `e250_unionr8_ens_v3` | `union_r8` | `--rotations-per-complex 8` | 66-73 |
| `e250_unionheavy_v3` | `union_heavy` | (none; heavy pocket at inference) | 74-81 |

Common: `--label-cap 13 --max-epochs 20 --early-stop-patience 4 --block-size 640`,
backbone `mlm_e250lig3` e02, ligand readout, f16 and f32 inference.

**The headline model is the z-sum of all 24**, no seed selection, no weighting,
no blending, no frame mixing.

## Claims (f32 is the basis; f16 reported beside)

* **Y1.** R(ours 24) > R(GenScore, 1 released checkpoint) AND
  rho(ours 24) > rho(GenScore, 1 released checkpoint) -- i.e. > 0.8127 and > 0.7281.
* **Y2.** R(ours 24) > R(GenScore, 3 released checkpoints) AND
  rho(ours 24) > rho(GenScore, 3 released checkpoints) -- i.e. > 0.8387 and > 0.7211.
* **Y3.** R(ours 24) > 0.83.

Reported beside, not part of the verdict: twin-clean columns (19-complex set,
n=266), our mean single-seed R/rho against GenScore's single checkpoint, and the
same mix built from the already-scored seeds (exploratory).

## Fixed now so it cannot be spun afterwards

* A Y1 pass with a Y2 fail is "beats one released GenScore checkpoint, loses to
  an ensemble of three", and must be written that way, with both rows.
* 24 models of ours against 1 or 3 of theirs is not equal cost. Every statement
  names both counts, and the single-model comparison is shown next to it.
* Ours uses no contact pairs, interaction edges, distance bins or contact
  counts. GenScore's score comes from learned protein-ligand distance
  distributions.
* If the fresh mix lands below the already-seen mixes, that is reported as the
  result; the seen numbers do not become the headline.

## Cost

24 heads (gpu_1, 1:00) ~= 4.8 pt + 48 inferences (gpu_1, 0:30) ~= 4.8 pt
~= **9.6 pt**, taking the ledger to about 149 of the 200 pt cap.

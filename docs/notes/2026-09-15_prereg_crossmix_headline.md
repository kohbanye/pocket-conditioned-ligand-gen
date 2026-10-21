# Pre-registration: headline for the two-recipe mix (fresh seeds)

2026-09-15 ~16:35 JST. Written while the 48 jobs of seeds 18-25 are running and
BEFORE any CASF number for those seeds exists. The lever verdict (M1/M2) is
pre-registered separately in `docs/notes/2026-09-15_prereg_cross_recipe_mix.md`.

## What is reported

**Ours:** the z-sum of the **16 fresh models** -- `e250_unionnc_corr_v2` seeds
18-25 plus `e250_unionr8_ens_v2` seeds 18-25. Fresh seeds only; the 16
already-scored models of the same two recipes are exploratory and never the
headline, because their mix was seen before this note was written.
f32 is the basis, f16 reported beside.

**GenScore, both references** (`docs/results/2026-09-15_genscore_ensembles_and_the_unionr8_headline.md`):

1. single released checkpoint `GatedGCN_ft_1.0_1.pth`: R 0.8127, rho 0.7281;
2. **3 released checkpoints z-sum: R 0.8387, rho 0.7211** -- the like-for-like
   reference for an ensemble, and the one the headline leads with.

## Claims (f32)

* **X1.** R(ours 16) > R(GenScore 3) AND rho(ours 16) > rho(GenScore 3).
* **X2.** R(ours 16) > R(GenScore 1) AND rho(ours 16) > rho(GenScore 1).
* **X3.** R(ours 16) > 0.83.

Reported beside: twin-clean columns (19-complex set, n=266), our mean single
seed against GenScore's single checkpoint, and the M1/M2 lever verdict.

## Fixed now, so it cannot be spun later

* A pass on X2 with a fail on X1 is "beats one GenScore checkpoint but not an
  ensemble of three", and must be written that way, with both rows shown.
* 16 of ours against 3 of theirs is not equal-cost either. Any claim states
  both counts, and the single-model comparison is shown next to it.
* Ours uses no contact pairs, interaction edges, distance bins or contact
  counts; GenScore's score comes from learned protein-ligand distance
  distributions.
* No seed selection, no weight tuning, no blending: the mix rule is "every
  fresh seed of both recipes, z-sum".

## Cost

None beyond the 6.4 pt already committed to the fresh seeds.

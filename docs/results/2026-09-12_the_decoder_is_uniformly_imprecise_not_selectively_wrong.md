# The decoder is uniformly imprecise, not selectively wrong

Fine-tuning the decoder realises 2% of the 0.816 kcal its own floor is worth and
plateaus at epoch 9 of 60; 12 layers beat 6 by 0.016 A and 18 is worse than 12.
So the limit is design, not optimisation — and what to change depends on the
*shape* of the error, which only the molecule-level RMSD had ever been measured
at.

2275 reference-ligand atoms over 79 clean targets, tokenizer alone (the
molecule's own codes, no language model):

| quantile | per-atom displacement |
|---|---|
| 10% | 0.168 A |
| 25% | 0.245 A |
| **50%** | **0.366 A** |
| 90% | 0.717 A |
| 95% | 0.917 A |
| 99% | 1.443 A |
| max | 3.222 A |

**mean / median = 1.15.** The worst 10% of atoms carry 48% of the squared error,
which is what a mildly skewed unimodal distribution gives; it is not a tail of
broken atoms sitting on top of a good model.

## Nothing predicts it

| predictor | Spearman rho | p |
|---|---|---|
| distance from the molecule's own centroid | **+0.257** | 1.4e-35 |
| bonded degree | −0.186 | 4.2e-19 |
| burial (receptor atoms within 6 A) | −0.173 | 1e-16 |
| molecule size | +0.161 | 9.6e-15 |
| position along the decode order | +0.104 | 6e-07 |

Every correlation is significant and none explains more than about 7% of the
variance. By degree: **1 → 0.428, 2 → 0.366, 3 → 0.311** — the periphery is
worst, which is the same gradient as distance from the centroid. By element the
whole spread is 0.08 A (C 0.337 best, Cl 0.613 worst on n=11).

**So the decoder is imprecise roughly everywhere, slightly more so at the
molecule's edge.** There is no subclass of atom to fix, which rules out targeted
architectural changes aimed at one.

## What this does to the redesign options

It **weakens the case for internal-coordinate decoding.** The argument for it was
that bond lengths come back at 0.058 A MAE while the `bond12` loss carries weight
5.0, so the auxiliary losses look like they are fighting the head. Put beside the
number above, they are fighting it and **winning**: local geometry is already
**6x more accurate than absolute position** (0.058 against 0.366). Making bonds
and angles exact by construction would guarantee the part that is already good,
and internal coordinates accumulate along the chain, which would make the part
that is bad worse.

The decoder's problem is **placing the local cluster, not shaping it.**

That points at the options that address placement:

* **condition the decoder on the pocket** — it currently maps codes to
  coordinates with no idea where the protein is, while the clash probe, the
  pruner and Vina all see the receptor. Testable without touching the encoder,
  since `--freeze-encoder` is verified to leave the codes bit-identical;
* **an explicit placement head** — one global transform separate from the local
  shape, matching the 84% / 16% split measured in
  `2026-09-12_the_shape_is_right_the_placement_is_not.md`, instead of entangling
  both in per-atom absolute coordinates and bolting a refiner on afterwards.

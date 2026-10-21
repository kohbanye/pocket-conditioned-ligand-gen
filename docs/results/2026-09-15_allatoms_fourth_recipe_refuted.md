# A fourth recipe (all-atom readout): refuted at fixed ensemble size

2026-09-15 ~17:35 JST. Pre-registered in
`docs/notes/2026-09-15_prereg_allatoms_fourth_recipe.md` before any head
existed, including the note that this axis was expected to be weaker than the
heavy one. Applied by `allatoms_verdict.py`. All 24 jobs exited 0; every head
ran `--readout allatoms` on `union_noconf` at 479 steps/epoch (run.json).

## Result (f32, 24 models on both sides)

| | n | R | between | within | rho |
|---|---|---|---|---|---|
| fresh mix (2 recipes x 8) | 16 | 0.8152 | 0.3915 | 0.4237 | 0.7123 |
| `e250_unionallatoms` alone | 8 | 0.8064 | **0.4032** | 0.4033 | 0.6772 |
| **TREAT** fresh 16 + allatoms 8 | 24 | 0.8174 | 0.3979 | 0.4194 | 0.7228 |
| **CTRL** fresh 16 + same-recipe 8 | 24 | 0.8159 | 0.3879 | 0.4280 | 0.7228 |

**A1: dR +0.0014 < +0.004 FAIL. A2: drho +0.0000 PASS. Lever DOES NOT WORK**
(both were required). At f16: dR +0.0016, drho -0.0088.

Residual correlation of the new family against the two mixed recipes: **+0.962**
(plain) and **+0.890** (rotation) -- matching the dev-cohort measurement quoted
in the pre-registration, and the reason the expected gain was stated as small
before the round ran.

## Reading

* The all-atom readout has the **highest between-target R of any family here**
  (0.4032, above the rotation recipe's 0.3971) and the lowest within (0.4033).
  The profile is right; the diversity it adds beyond the rotation recipe is not,
  because rotation already occupies that corner (their residuals correlate
  0.890, the least of any pair we have, yet the mix gains almost nothing).
* Two recipes with the same *kind* of complementarity do not stack. The
  confirmed lever was one plain + one between-heavy recipe; adding a second
  between-heavy recipe is close to adding more seeds.
* The 24-model treatment reaches R 0.8174 / rho 0.7228 at f32 and R 0.8181 /
  rho 0.7281 at f16 -- the rho at f16 exactly ties GenScore's single checkpoint,
  which is not a win. R still trails GenScore's 3-checkpoint ensemble (0.8387)
  by 0.021.

## Cost

8 heads + 16 inferences, ~3.2 pt. Ledger 131.31/200.

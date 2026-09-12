# Widening the picker's window changes nothing

`--iter-candidates` controls how many of the model's most probable codes the
clash-aware picker checks for a replacement. It is 256 by default and had never
been swept. The case for raising it was measured: of the escape codes available
to a terminal atom that still clashes after clash-ordered decoding, only **60.5%
sit within the top 256**, and 77.6% within 1024 — so two fifths of them are
invisible to the picker.

Run at 1024 against the identical stack at 256, 24 targets, same seed.

| arm | score | min | PB | strain | clashes | heavy | unique |
|---|---|---|---|---|---|---|---|
| candidates 256 | −0.450 | | 0.526 | 559 | 2.10 | 23.31 | 0.566 |
| candidates 1024 | −0.511 | −5.228 | 0.523 | 561 | **2.07** | 23.31 | 0.566 |

paired `vina_score`: **+0.000**, 1024 better on 29% of targets.

Clean 18 of those targets: +0.941 against +0.861, paired **+0.000** again.

**The molecules are the same molecules.** Heavy-atom count and uniqueness agree to
three digits, PoseBusters to three digits, and the clash count moves by 0.03 per
molecule. The flag was applied — the run records show
`--iter-order clash --iter-candidates 1024` against the control's default.

## Why, and what it closes

The escape rank measurement was right and the inference from it was wrong. The
atoms it describes are the residual **after** clash ordering: about one per
molecule. Widening the window shows the picker more codes for those atoms, and it
still does not take them, because it picks the most probable code that clears and
**the probability mass on escapes is 0.000 at the median**. A code at rank 800 is
visible at 1024 and still never the most probable clear option.

So the 75.8% of residual clashing atoms that had an escape and did not take it
are not a window problem. What is left is the accept probe — the bond-length,
angle and non-bonded windows that a replacement must also satisfy. Those atoms
are ones where clearing the wall would break local geometry, which is a real
trade and not a defect: removing those checks cost 7.0 points of PoseBusters when
it was tried.

**The clash-ordering axis is saturated**, consistent with 3x the rounds being
worth 0.032 kcal. Neither more rounds, nor more candidates, nor a finer codebook
(granularity binds 9.9% of the residual) moves it.

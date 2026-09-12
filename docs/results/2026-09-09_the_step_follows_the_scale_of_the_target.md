# The refiner's step length follows the scale of its training target — which explains four separate failures

2026-09-09. Registered in
`docs/notes/2026-09-09_corruption_rotation_cap_prereg.md`. The registered
mechanism check failed **in the opposite direction to the prediction**, and
that failure is the useful result; the score merely confirms it.

## The arm

`refiner_rotcap30`: `refiner_overlap_t0` plus `--max-corrupt-rot-deg 30`,
dropping the 36% of training records whose manufactured corruption rotates the
ligand further than any deployment pose needs (25.4% of the corpus exceeds 45°,
13.8% exceeds 90°, while a generated pose needs 18.4–18.6°). Same corpus, same
overlap target, same t≡0 schedule, same warm start, same 6 epochs, lr and seed.
The filter is a view: `tests/test_refine_projection.py` asserts the survivors
come back byte for byte.

## The mechanism check, read before the score

| | `refrotcap30` | `refoverlap_t0` |
|---|---|---|
| **rigid rotation, median** | **3.715°** | 3.964° |
| displacement RMSD, median | 0.701 Å | 0.754 Å |
| centroid translation, median | 0.641 Å | 0.688 Å |

Registered: the median rotation must **rise above 3.96°**. It fell.

The prereg's reasoning was that the conditional mean of a multimodal *large*
rotation is short, so removing the tail should let the step lengthen. Wrong.
Removing the tail also lowered the mean **magnitude** of the training targets,
and the step followed that down.

## The account, checked quantitatively

| | targets the control saw | targets the arm saw | ratio |
|---|---|---|---|
| overlap-target move, median RMSD | 1.961 Å | 1.751 Å | **0.893** |
| the net's own step, median RMSD | 0.754 Å | 0.701 Å | **0.930** |

The target shrank by 0.89 and the step by 0.93. (The rotation ratios, 0.66
against 0.94, do not line up, and should not: the control's target rotation
sits at exactly **30.00°** at the median — the search cap, saturated, as
recorded on 2026-09-07. Ratios of a saturated quantity mean nothing.)

**The step length is a roughly fixed fraction of the target's magnitude**
— 0.75 / 1.96 = 0.38 for the control, 0.70 / 1.75 = 0.40 for the arm.

## Why this matters more than the arm

It collapses four independently-run failures into one mechanism:

| what was changed | outcome | why, under this account |
|---|---|---|
| α-scale the output (`refiner_gain_probe`) | flat to 1.5×, collapses from 2× | changes the output's scale without changing what training taught the net to expect |
| more rounds (`refoverlap_t0r2`) | null, p=0.52 | same |
| a better teacher (`refiner_lj_t0`) | **−0.276, p=5.7e-09** | changes the direction, leaves the scale, so the step does not lengthen |
| cut the corruption's tail (this arm) | **−0.168, p=3.7e-05** | **lowers the scale, and the step shrank with it** |

The refiner's step is not short because the target is wrong, nor because the
head is wrong, nor because the model hedges across modes. It is short because
it is a fixed fraction of what it was shown, and what it was shown is a target
whose median move is 1.96 Å.

Lengthening it therefore needs a target that is genuinely larger — but the
target is the bounded transform that relieves the receptor overlap, a
deterministic geometric quantity that cannot be inflated without ceasing to be
the thing the score responds to.

## Score

| basis | `refrotcap30` | `refoverlap_t0` | paired | p | arm better |
|---|---|---|---|---|---|
| vina_score, 100 | −0.341 | **−0.504** | **+0.168** | **3.7e-05** | 27% |
| vina_score, clean 79 | 0.565 | **0.470** | **+0.189** | **1.1e-05** | 23% |
| vina_min, 100 | −4.791 | −4.825 | +0.013 | 0.520 | 46% |

Registered: −0.2 to −0.6 kcal improvement.

## Costs, co-reported — structurally zero as registered

| | `refrotcap30` | `refoverlap_t0` |
|---|---|---|
| uniqueness | 0.5469 | 0.5470 |
| scaffold diversity | 0.3340 | 0.3340 |
| novelty | 0.8202 | 0.8202 |
| validity | 0.9940 | 0.9940 |
| PoseBusters validity | 0.5722 | 0.5723 |
| strain | 856.8 | 858.9 |
| clash-free | 0.3100 | 0.3213 |

Same LM, same seed, same molecules; only where they sit differs. Not
disqualified.

The confound the prereg declared — that filtering also removes 36% of the
records — **does not need its control run**. The arm lost, so there is nothing
to attribute, and the mechanism check had already falsified the reasoning
before the score arrived.

## The refiner axis, final

| what was changed | outcome |
|---|---|
| target = crystal pose | loses, p=8.9e-04 |
| target = LJ well | loses, p=5.7e-09 |
| target with rotation removed | loses, p=0.035 |
| head = rotation / torsion | does not converge |
| step magnitude (α) | flat, then collapses |
| step count (rounds) | null |
| training-input rotation cap | **loses, p=3.7e-05** |
| **schedule t≡0** | **the one that works, −0.482, p=4.0e-03** |

Eight settings, one works, and the deployed configuration is it.

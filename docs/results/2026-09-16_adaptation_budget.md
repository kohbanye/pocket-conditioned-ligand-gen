# Constraining encoder adaptation: LoRA earns something, but trades between for within

2026-09-16 ~00:30 JST. Pre-registered in `docs/notes/2026-09-15_prereg_lora.md`
(rewritten before submission, after checking the checkpoints showed every arm in
this loop fine-tunes the full ~105M encoder). Applied by `adapt_verdict.py` via
an unattended collector job. All 48 jobs exited 0.

## The adapters did train

Mean |B| per seed 5.8e-03 to 1.25e-02, mean **9.05e-03** against the
pre-registered void floor of 1e-04. The round tests a real adaptation, not a
no-op.

## Result (f32 is the pre-registered basis)

| | n | R | between | within | rho |
|---|---|---|---|---|---|
| **CTRL** full fine-tune (`corr_v3`) | 8 | **0.8129** | **0.3761** | 0.4369 | **0.7105** |
| **B** frozen + LoRA rank 4 | 8 | 0.7473 | 0.2842 | **0.4631** | 0.6228 |
| **A** frozen encoder, head only | 8 | 0.7291 | 0.2723 | 0.4568 | 0.6158 |

**P1 (B beats CTRL on R): FALSE** (-0.0656). **P2 (rho): FALSE** (-0.0877).
**P3 (B beats A on R): TRUE** (+0.0182). At f16: -0.0646, -0.0965, +0.0165.

## What it shows

* **LoRA works, and is not enough.** ~0.4M adapter parameters recover +0.0182 R
  over head-only — **22% of the 0.0838 gap** to full fine-tuning, and only
  **11%** of the between-target part of it (0.0119 of 0.1038). Most of what
  updating 105M parameters buys cannot be bought with a low-rank update.
* **Adaptation budget trades within for between.** Both constrained variants
  *beat* full fine-tuning on within-target ranking (0.4631 and 0.4568 against
  0.4369) while losing badly on between. Full fine-tuning buys cross-target
  scale by giving up within-target discrimination.

## The mixing follow-up also fails (exploratory, size-matched)

The within-strong profile suggested adding these families to the confirmed
three-recipe mix. Against a **size-matched 32-model control** (24 + 8 more seeds
of existing recipes, R 0.8230 / rho 0.7333 at f32):

| 32-model mix | dR vs control | drho vs control |
|---|---|---|
| + LoRA 8 | **-0.0022** | -0.0123 |
| + frozen 8 | **-0.0036** | -0.0175 |

The profile shift is real (within 0.4377 → 0.4530, between 0.3853 → 0.3678) but
the between loss outweighs the within gain. The mix is not short of within.

With the all-atom round (between-heavy, refuted) this closes the confirmed
mixing lever in **both** directions: neither a more-between nor a more-within
recipe adds to it.

## Cost

16 heads + 32 inferences + collector, ~9.7 pt. Ledger 173.78/200.

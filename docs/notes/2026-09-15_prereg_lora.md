# Pre-registration: how much encoder adaptation does this task want?

2026-09-15 ~23:20 JST. Written BEFORE any head of these arms exists. **This
note replaces an earlier draft of itself that was factually wrong**; the
correction is recorded below rather than quietly edited away.

## The correction that forced a redesign

The first draft proposed LoRA as "more adaptation than the frozen baseline" and
claimed full fine-tuning had been refuted. Checking the actual checkpoints
before submitting showed the opposite:

* `RescoreTrainingConfig.freeze_encoder` defaults to **False**, the
  `--freeze-encoder` flag is opt-in, and **no head script in this loop passes
  it**.
* A trained control checkpoint (`aff_e250_unionnccorr_s58`) records
  `freeze_encoder = False` and carries `encoder 105.3M + head 0.59M`.

So every arm in this loop -- including the 36-model mix that beats GenScore's
single checkpoint -- **fine-tunes the whole ~105M encoder end to end**. What a
dev diagnostic refuted earlier was `encoder_lr_scale` (a *scaled* encoder
learning rate), not fine-tuning itself.

That makes "frozen + LoRA" **less** adaptation than our baseline, not more, and
turns the question into one worth asking: with only ~15k training complexes and
a 285-complex test set, does full 105M-parameter adaptation overfit relative to
a constrained one? Schmirler, Heinzinger & Rost (*Nat. Commun.* 2024) found PEFT
competitive with or better than full fine-tuning across protein tasks, with LoRA
the best of four PEFT methods -- though their regression tasks often did not
separate within 95% CI, and none is protein-ligand affinity.

## Arms (three points on one axis: how many parameters may adapt)

| arm | encoder | trainable | status |
|---|---|---|---|
| **control** `e250_unionnc_corr_v3` | full fine-tune | ~105.9M | trained, scored |
| **A** `e250_unionnc_frozen` (NEW, seeds 110-117) | frozen, eval mode | 0.59M (head only) | this round |
| **B** `e250_unionnc_lora` (NEW, seeds 118-125) | frozen + LoRA rank 4 on every attention block's QKV and output projections | 0.59M + ~0.4M adapters | this round |

Everything else is the standard recipe and identical across the three:
`union_noconf`, `--corr-weight 0.7 --label-cap 13 --max-epochs 20
--early-stop-patience 4 --block-size 640`, ligand readout, f16/f32 inference.

LoRA detail: `base(x) + (alpha/rank)*B(A(x))` with `B` initialised to zeros, so
a freshly wrapped encoder is bit-identical to the frozen one and only training
moves it (fixed by `tests/test_lora.py`).

## Predictions (f32 basis, 8-seed family ensembles)

* **P1.** R(B, frozen+LoRA) - R(control, full FT) >= **+0.005**.
* **P2.** rho(B) >= rho(control) - 0.005.
* **P3 (the isolating comparison).** R(B) - R(A, frozen only) >= **+0.005** --
  i.e. the adapters, not merely freezing, are what moved it.

**"LoRA works" requires P1 and P2. P3 says whether the adapters earned it.**
If A alone beats the control, the finding is "full fine-tuning overfits here",
which is reported as such and is arguably the more useful result.

**Void condition:** if B's trained `B` matrices stay at zero (mean |B| < 1e-4),
the adapters never engaged; reported void, not refuted.

## Cost

A: 8 heads (gpu_1, 1:00, head-only so fast) ~= 1.6 pt + 16 inferences ~= 1.6 pt.
B: 8 heads (gpu_1, 3:00, backprop reaches the frozen encoder) ~= 4.8 pt + 16
inferences ~= 1.6 pt. Two collectors ~= 0.06 pt. **~9.7 pt**, ledger to about
170 of the 200 cap.

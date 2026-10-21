# Averaging every encoder layer is worse than reading the last one

2026-09-15 ~21:15 JST. Pre-registered in
`docs/notes/2026-09-15_prereg_layer_mix.md` before the option was implemented.
Applied by `layermix_verdict.py`. All 24 jobs exited 0.

## The weights never learned, so this is not the test that was intended

The mixing weights were initialised to zeros (uniform softmax) and **stayed
there**: across all 8 seeds every layer holds 0.071-0.073 against a uniform
1/14 = 0.0714. The largest single weight in any seed is the embedding output at
0.073. Fourteen scalars behind a frozen encoder, trained at the head's learning
rate, received too little gradient to move.

So the round measured **"the uniform average of the embedding output and all 13
layers" against "the last layer"** -- not "a head that learns which layer to
read", which is what the pre-registration described. That is recorded here
rather than reinterpreted after the fact.

## Result (f32 is the pre-registered basis)

| | n | R | between | within | rho |
|---|---|---|---|---|---|
| **TREAT** `e250_unionnc_layermix` | 8 | 0.7957 | 0.3628 | 0.4329 | 0.6825 |
| **CTRL** `e250_unionnc_corr_v3` (last layer) | 8 | 0.8129 | 0.3761 | 0.4369 | 0.7105 |

**Z1: dR -0.0173 FAIL. Z2: drho -0.0281 FAIL. LEVER DOES NOT WORK.**
At f16: dR -0.0165, drho -0.0211.

Swapping the family into the 24-model headline mix costs it too: R 0.8191 /
rho 0.7123 against the pre-registered mix's 0.8244 / 0.7228.

## Reading

* Averaging in the embedding output and the early layers **dilutes the
  contextualised signal**. The damage is largest in the between-target part
  (0.3761 -> 0.3628), which is the half that needs the pocket context the deep
  layers build.
* This does **not** refute "an intermediate layer is better than the last one".
  That question is untouched, because the head never got to choose.
* The infrastructure is verified end to end: the option adds parameters only
  when enabled, a layer-mix checkpoint round-trips through
  `load_scoring_head(layer_mix=True)`, and its dumps are sane (285 rows, single
  seed R 0.805 at f16).

## Two bugs worth keeping

* `torch.load` in the verdict's weight reader hit torch>=2.6's
  `weights_only=True` default, because our checkpoints pickle the config
  instance. **The smoke test passed anyway** -- it returned early on missing
  dumps and never reached that code.
* A check that reports "0 problems" without having executed its loop body is
  worse than a failing check (`set -- $var` does not split in zsh). Both are now
  in the feedback memories.

## Next

`docs/notes/2026-09-15_prereg_layer_mix_learnable.md`: initialise the weights
favouring the last layer (so the run *starts* at the control's behaviour) and
give them their own learning rate, with a pre-registered void condition if they
still do not move.

## Cost

8 heads + 16 inferences, ~3.2 pt. Ledger 152.12/200.

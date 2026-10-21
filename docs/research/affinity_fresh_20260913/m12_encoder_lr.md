# M12 encoder learning-rate scale

Opt-in: append `--encoder-lr-scale 0.1` or `--encoder-lr-scale 0.3` to the same scoring-head training command. Head LR remains the existing `--lr`/default value; encoder LR is multiplied. No architecture, loss, readout, data, or default changes.

For scale1.0, the original parameter order, two decay groups, AdamW defaults and LinearLR→CosineAnnealingLR schedule are untouched. For scale<1, trainable encoder/head parameters are partitioned separately with the same decay policy. A shared LambdaLR multiplier maintains the ratio at warmup start, peak and minimum LR. Frozen encoder parameters are excluded. Values outside(0,1], includingNaN, fail at model construction.

Tests: `tests/test_rescore_encoder_lr.py` verifies exact default all-step LR compatibility against the original recipe, default parameter ordering, no duplicate/missing trainable parameters, decay policy, 0.1/0.3 schedules (including frozen encoder), endpoint scaling and invalid values. Together with existing config contract tests,11passed. Targeted ruff passed. No job submissions.

A screen must keep backbone checkpoint, corpus, microbatch/steps, head LR, split, seeds and evaluation policy fixed, changing only the scale. This tests adaptation strength of the representation, not extra prediction-head capacity. Lowering all parameter learning rates is a different experiment. Match eventual scheduled step budgets explicitly: a fixed epoch count and early stopping can otherwise yield different optimization exposure.

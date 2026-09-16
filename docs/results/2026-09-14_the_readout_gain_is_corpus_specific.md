# The readout gain reverses sign on a different corpus

2026-09-14, evening. This morning the M14 readout change improved ranking rho
on 4/4 seeds of the `ca_legacy` development cohort (+0.0171), and I recorded
the interpretation that a readout benefit is "a property of the representation,
which is the kind of thing that travels." Put on a different corpus, it does
not travel. It reverses.

## The measurement

`union_noconf` is the corpus that leads on scoring R (family mean 0.7971).
Heads were retrained on it with exactly the `aff_e250_unionnc_s*` recipe and
only `--readout` changed, four seeds each, then scored on CASF at 285-complex
coverage. Control is `e250_union_noconf` itself -- same corpus, same recipe,
ligand readout.

| vs control | n_better R | n_better rho | mean ΔR | mean Δrho |
|---|---:|---:|---:|---:|
| `union_noconf` × `allatoms` | 1/3 | **0/3** | −0.0087 | **−0.0515** |
| `union_noconf` × `balanced` | 1/3 | **0/3** | −0.0272 | −0.0561 |

Per seed, `allatoms`: −0.0807, −0.0474, −0.0263 rho. Not one seed improves.

The control family has no seed 13, so the paired comparison runs on seeds
7/11/12. That does not soften the verdict: 0/3 with every delta negative and
the smallest of them −0.026.

## Same readout, same tokenizer, opposite sign

| corpus | Δrho vs its own control | n_better |
|---|---:|---|
| `ca_legacy` (development cohort) | **+0.0171** | 4/4 |
| `union_noconf` | **−0.0515** | 0/3 |

The tokenizer, the MLM, the head architecture, the parameter count and the
pooling arithmetic are identical across those two rows. Only the training
corpus differs. So the readout is not a property of the representation that a
better pooling unlocks; whatever it did on `ca_legacy` was specific to that
corpus.

## Correction to this morning's note

`2026-09-14_the_readout_transfers_the_inputs_did_not.md` argued that the input
repair failed to transfer because it shifted the input distribution, while the
readout succeeded because it touched only which states are averaged, and
concluded "the protein rows carry usable affinity signal." That conclusion is
**withdrawn**. It was drawn from one corpus, and the second corpus contradicts
it. The honest version: one dev effect failed to transfer to CASF, and the
other transferred on the cohort it was measured on but not to a different
training corpus. Neither is a mechanism.

The general lesson this repository keeps relearning applies here too: an effect
measured in one configuration is a statement about that configuration until it
is reproduced in another.

## Standings, unchanged in the only way that matters

Best single arm of the new families is `balanced` seed 7 at R 0.8098 / rho
0.6754. Its R is within 0.003 of GenScore; its rho is 0.053 short. **No arm,
at any seed, beats GenScore on both metrics** -- still true across all 59
registered arms.

## What is left

Nothing on the identified list. Every affinity lead that had a measured effect
has now been carried to CASF and has failed there:

- corpus union (H7), contradictory labels (H8), listwise (H9, three times)
- input repair: dev +0.030 → CASF −0.0015
- readout: dev +0.017 → reverses to −0.05 on another corpus

The two metrics are won by different corpora (R by `union_noconf`, rho by
`rot1`), consistently, and no intervention has moved both together. Continuing
to sample this space without a mechanism is search, not investigation, and it
should be a deliberate choice rather than a default.

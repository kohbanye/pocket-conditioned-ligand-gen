# The decoder holds more of the round trip than the codebook, and it is free

Three ways to make the tokenizer more faithful, measured against each other on
the **dfs** tokenizer (`vq_ord_buriedfirst`), canonical 100.

## Why this matters at all, given the ceiling is not binding

The best arm scores −1.308 and the dfs ceiling is −5.496, so there is 4.19 kcal
of headroom and **raising the ceiling gains nothing today**. The reason to do it
anyway is the target: the reference ligand is −6.871 and FLOWR is −5.730.

**No model on this tokenizer can ever pass −5.496.** The stated goal is to
approach −6.806. That is not reachable here, and even FLOWR is only passed once
the refiner is added. Tokenizer work buys zero now and is the only thing that
can ever lift that cap.

## Where the round trip goes (`quantisation_vs_decoder.py`, n=100)

The same latents decoded twice — once through the codebook, once directly.

| | RMSD | kcal at 3.69 kcal/A |
|---|---|---|
| through the codebook (the deployed path) | 0.3902 A | — |
| **bypassing it — the decoder's own floor** | **0.2212 A** | **0.816** |
| the difference — all a bigger book could return | 0.1567 A | **0.578** |

Latent quantisation error is 21.2% of the latent norm.

**The decoder holds the larger half.** The published tokenizer split the same way
(0.350 / 0.187 / 0.147); dfs is slightly worse throughout, and the shares hold.

## What a bigger book would actually buy (`codebook_size_scaling.py`)

22,523 held-out latents, dim 16, k-means at three sizes.

| k | flat error | vs the deployed 8192 |
|---|---|---|
| 256 | 10.0446 | 2.604x |
| 1024 | 7.2911 | 1.890x |
| 4096 | 5.0353 | 1.306x |

`err ~ k^-0.2491`, an **effective dimension of 4.0** — so 4x the codes buys 29.2%
less error and 8x buys 40.4%. The textbook `k^(-1/d)` on a 16-dimensional latent
would have predicted far worse; the latent's real support is 4-dimensional.

A second stage over the first book's residual does much better:

| 2nd-stage k | residual after it | vs deployed |
|---|---|---|
| 256 | 1.7885 | 0.464x |
| 1024 | 1.5153 | 0.393x |
| 4096 | 1.3247 | 0.343x |

## The three routes, priced

| change | ceiling gain | what it costs downstream |
|---|---|---|
| **decoder, perfect** | **0.816** | **nothing — the token stream is bit identical** |
| residual VQ, 4096 second stage | ~0.38 | every LM retrained, 2 tokens per atom |
| flat book, 8x | ~0.23 | every LM retrained, 8x vocabulary |

The first number is an unreachable bound (a perfect decoder), but even a third of
it beats both codebook routes, and it is the only one that does not invalidate
`data/lm_tokens_*` and every CLM checkpoint built on them.

**The cheapest route is also the biggest.** That is not the order the record
assumed: `2026-09-02_the_codebook_holds_half_a_kcal.md` licensed "residual VQ /
codebook growth" on a measured 0.54 kcal and set the decoder aside as "a
different axis". It is a different axis, and it is the larger one.

## Why the decoder route is safe, mechanically

`--freeze-encoder` freezes the encoder **and the codebook as one unit** — the
module's `_ENCODER_PARTS` includes `codebook`, and the codebook is held in
`eval()` rather than merely `requires_grad=False`, because its EMA update runs
inside its own forward gated on `self.training`. `train()` is overridden so
Lightning putting the module back in train mode each epoch does not undo it.
Freezing the encoder while letting the codebook drift would change the codes,
which is the one thing this must not do.

So `descriptor -> code` is unchanged and only `code -> coordinates` moves.

## What is running

`--init-from <dfs last> --freeze-encoder`, 40 epochs, on the cache the tokenizer
was trained on (`descriptor_cache_ord_good`; `ord_full`'s normalization
statistics differ in the 4th decimal and pairing the wrong ones rescales the
coordinates without raising).

Two arms, because one run would confound two questions:

* `vq_ord_dec` — published recon weights. Asks whether a 3.24M-parameter decoder
  that already saw 97,250 steps has any coordinate accuracy left, or is
  saturated and needs to be **larger** rather than trained longer.
* `vq_ord_decbal` — `--balanced-chem-loss`. The aromatic head answers "not
  aromatic" almost always (recall 0.014) and inverse-frequency weighting is the
  designed fix. Never used before either.

Both flags existed and neither had ever been run.

**The token stream must be verified bit-identical after training, not assumed.**

# Three tokenizers, three ceilings, measured the same way

The crystal ligand encoded and decoded through each tokenizer, scored exactly as
a generated pose. No language model: this is what a **perfect** one could score.
Clean 79 basis, one script, one run each.

| tokenizer | ceiling | RMSD | bond MAE | best arm on it | that arm's score | headroom |
|---|---|---|---|---|---|---|
| published (HF blobs) | **−5.550** | 0.362 | 0.043 | `flow3np` (previous generation) | −0.969 | 4.581 |
| dfs / `vq_ord_buriedfirst` | **−5.496** | 0.402 | 0.058 | `+` MLM `+` clash order | −0.794 | 4.702 |
| **pctx / `pctx_final`** | **−3.519** | 0.876 | **0.201** | clash+geom `+` pruning | **−1.403** | 2.115 |
| crystal reference | −6.871 | — | — | | | |
| FLOWR (recorded, same evaluator) | −5.730 | | | | | |

**Two tokenizers agree at about −5.5 and pctx is 2.0 kcal below them**, with bond
lengths coming back 4x less accurately.

## What this does to the reading of the gap

The best arm's remaining 5.468 kcal was being attributed to placement. On the
pctx stack most of it is not:

| component | kcal | share |
|---|---|---|
| the tokenizer's round trip | 3.352 | **61%** |
| the model and the placement | 2.116 | 39% |

## The training metric says the opposite

The two tokenizers were trained with **identical commands** — same flags, same
seed, same 250 epochs — differing only in the descriptor cache. And pctx has the
*better* validation coordinate loss at the same epoch: `atom_coord=0.1208`
against `0.1515`. The normalisation stats are the same to four decimals
(coord std 2.9276 vs 2.9275), so the two losses are in the same units.

A tokenizer that wins on its own validation metric and reconstructs 2x worse is
the same failure already recorded for the pose refiner: **do not rank by the
learning curve**.

## The objection that had to be tested first

`--pocket-context` widens each ligand atom's knn *search* set to the pocket, and
the descriptor stays 33-D, so a tokenizer trained on such a cache can be fed a
plain descriptor with **no error raised**. The pctx ceiling above was measured
with a plain descriptor, which would invalidate it.

Tested on 12 targets, same tokenizer, descriptor built both ways:

| descriptor | RMSD | bond MAE | round-trip score |
|---|---|---|---|
| plain | 1.022 | 0.316 | −3.875 |
| with pocket context | 1.017 | 0.337 | −4.221 |

**No difference.** Feeding it the descriptor it was trained on does not rescue
the reconstruction, so the ceiling stands.

## The bug that made the objection possible

The builder's help says "*The cache records this choice; do not mix*". It did
not: `shard_metadata.pt` was byte-identical between the two caches
(`descriptor_kind: atom, dim 33, count 351006` for both). Two caches that must
not be mixed were indistinguishable on disk.

Fixed: `_save_atom_shard_metadata` now writes `pocket_context`, and
`tokenizer_vina_ceiling.py` takes `--pocket-context` with a note that passing
the wrong one raises nothing and silently reports a 2 kcal worse ceiling.

## What follows

For any **score** claim, the pctx stack is disqualified by its ceiling: 2.2 kcal
below FLOWR, so a perfect model on it still loses. The dfs and published
tokenizers are both competitive with FLOWR at the ceiling and both have ~4.6 kcal
of headroom left.

The pctx stack still holds the best arm measured here (−1.403 against
`flow3np`'s −0.969) **and** much better chemistry (PoseBusters 0.767 against
0.501, strain 418 against 1140). That is worth reporting as what the decoding
work achieves; it is not worth building the score claim on.

## Where the 2 kcal does NOT go: the codebook

The obvious explanation is capacity — the descriptor is 33-D and the codebook
8192 either way, so if pocket context is being represented it must come out of
the coordinates. Measured directly, by encoding every reference-ligand atom with
each tokenizer and asking how tightly a code pins a position down:

| | dfs | pctx |
|---|---|---|
| atoms encoded | 2275 | 2275 |
| codes used (of 8192) | 1890 | 1835 |
| codes seen more than once | 326 | 342 |
| within-code coordinate spread, median | 1.565 A | **1.574 A** |
| ... atom-weighted mean | 1.810 A | 1.782 A |

**Identical.** The codes partition coordinate space just as finely in the
pocket-context tokenizer, so the loss is not in the code assignment. From
equally informative codes, the pctx decoder reconstructs twice as coarsely.

That leaves the decoder: with pocket context, dims 13–33 hold protein-neighbour
offsets, which are more variable than ligand ones and are part of the
reconstruction target, so the shared trunk has more to fit. That is a hypothesis
and it is not tested here.

**It does not need to be, to act.** Pocket context costs 2.0 kcal of ceiling and
buys nothing measurable: the clash rate is the same as the deployed arm's (16.6%
vs 16.6%, paired p=0.41) and the score effect was null. **Do not use it.**

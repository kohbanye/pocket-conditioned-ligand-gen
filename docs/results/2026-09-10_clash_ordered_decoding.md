# Clash-ordered decoding: the mechanism works

`refine_codes(order="clash")`, added 2026-09-10. The MLM re-decides exactly the
ligand positions whose atoms are inside the protein, and takes the most probable
replacement from the model's top `candidates` that clears the wall.

## Why this and not a mask during generation

`--iter-order clash` is a **second pass**. Masking clashing codes while the
causal model emits cannot work: the position the VQ decoder gives the newest
atom from a left prefix is a median **0.930 A** from where that atom ends up
once the block is complete (78% move more than 0.5 A, 46% more than 1.0 A,
n=6456 atoms over 300 molecules), against a control of 0.439 A for the atoms
already placed. That is the same size as the clash criterion's own margin, so
the mask would veto codes on the basis of a location the atom never occupies.
The prior record already had this as "decode-time clash masking is impossible in
principle"; it holds, for the emission-time version only.

With the whole ligand block on the table the decode is exact, and the probe sees
where the atoms actually are.

## The confound that made the first version do nothing

The first wiring measured **no effect**: 3 clashing terminal atoms before, 3
after. The probe was judging the raw decode while the file receives the
**refined** pose -- the refiner runs after the codes are settled and moves the
whole molecule. Removing the refiner showed the mechanism was fine all along:

| one target, 24 molecules | clashing terminal atoms | per terminal atom |
|---|---|---|
| confidence, no refiner | 3 | 2.6% |
| **clash, no refiner** | **0** | **0.0%** |
| confidence, with refiner | 3 | 2.6% |
| clash, with refiner (probe blind to it) | 3 | 3.9% |

`--refine-project rigid` means the refiner's output is a rigid move of its
input, so the move is recovered once per molecule with `rigid_between` and
applied to every candidate -- one refiner call per round instead of 256. With
that, **clash with refiner: 3 -> 0**.

## At scale

All 96 paired targets, pctx+MLM stack. The control is the identical command with
`--iter-order confidence` (`canon100_pctx_t0full_mlm`), so the causal draw, the
seed, the tokenizer, the refiner and the number of re-decoded positions are all
the same and **only which positions get re-decided differs**.

| arm | clashing terminal atoms / molecule | per terminal atom | heavy | terminal | ring atoms |
|---|---|---|---|---|---|
| confidence (control) | 0.896 | 16.9% | 22.8 | 5.31 | 11.8 |
| **clash** | **0.490** | **9.1%** | 22.8 | 5.37 | 11.6 |

Paired per target: **−0.350, p=7.7e-15, fewer on 83% of targets** — a 45%
reduction, at identical molecule size, terminal-atom count and ring content.

It does not reach zero because `--iter-rounds 2 --iter-frac 0.05` fixes at most
two positions per molecule, 16% of clashing atoms have no escape code at all,
and the rigid-move reuse is an approximation. Raising the rounds is free for
clean molecules — `order="clash"` ends the schedule as soon as nothing is in the
wall — but it would stop this being an order-only ablation, so it waits for the
Vina result.

## The control on disk is a valid partner

`canon100_pctx_t0full_mlm` was generated **before** `refine_codes` was
refactored to pull the position choice into `_select`, so the comparison rests
on that refactor being behaviourally inert on the `confidence` path. The unit
tests assert it; this checks the artefact. One target regenerated with the
current code, same seed, `--iter-order confidence`:

- **100/100 identical SMILES**
- max per-molecule coordinate difference **0.0001 A** (the SDF's last printed
  digit), median 0.000000, and **no** molecule differs by more than 0.001 A

The ten differing text lines out of 7539 are float non-determinism, not a
behaviour change.

## What the re-decode costs in identity

Molecule *k* of each dump is the same causal draw, so the difference is the
re-decode alone:

| | |
|---|---|
| identical SMILES to the control | **34.3%** |
| Tanimoto to the control's molecule | median 0.694 |
| coordinate RMSD | median 0.368 A, p90 0.998 |
| atom count changed | 8.6% |

This is not a surgical repair: two thirds of the molecules come back as a
different compound. The control re-decodes two positions as well, so "rewriting
happens" is held equal between the arms — but the arm's effect is "different
molecules that also clash less", not "the same molecules with the clashes
removed", and it has to be read that way.

**This is the mechanism, not the verdict.** A conclusion from a partial shard set
has reversed at full coverage in this project before, and the score is a
different question from the clash count. Vina, PoseBusters, strain and diversity
are being measured against the control now.

## The verdict: a large score win, and a chemistry cost that was my bug

100 targets, 10000 molecules each, paired per target.

| metric | confidence | clash | paired | p | better on |
|---|---|---|---|---|---|
| **vina_score** | +1.704 | **−0.636** | **−1.215** | 1.7e-16 | **90%** |
| clash count | 2.425 | 1.060 | −1.000 | 1.2e-13 | 71% |
| PoseBusters valid | 0.805 | 0.718 | −0.070 | — | 6% |
| strain energy | 346 | 503 | +28.8 | 8.7e-10 | 23% |
| QED | 0.439 | 0.419 | −0.010 | 3e-08 | 10% |
| SA | 3.549 | 3.875 | +0.153 | 1.9e-12 | 4% |
| heavy atoms | 22.30 | 22.28 | 0.000 | 0.16 | — |
| molecular weight | 328.9 | 328.6 | 0.000 | 0.5 | — |

Score by target median: **−0.005 → −1.246**.

### Where it sits against everything else

| arm | vina (median) | vina (mean) | PB | strain | heavy |
|---|---|---|---|---|---|
| `refoverlap_t0` (deployed) | −0.504 | +1.303 | 0.572 | 859 | 22.3 |
| `+` clash-leaf pruning | −1.240 | −0.796 | 0.584 | 819 | 21.3 |
| pctx+MLM (confidence) | −0.005 | +1.704 | 0.805 | 346 | 22.3 |
| **pctx+MLM (clash order)** | **−1.246** | **−0.636** | 0.718 | 503 | 22.3 |

It **dominates the deployed arm on every axis at once** — better score, better
PoseBusters, much lower strain, same size — and matches the pruned arm's score
without shedding an atom.

### The chemistry cost is entirely one omission

The picker chose "the most probable code that clears the wall". The measurement
that licensed the design counted escapes that clear the wall **and keep the
element and the bond**. The implementation dropped the second half.

| | confidence | clash |
|---|---|---|
| molecules with a bond outside 1.10–1.95 A | 6.5% | **13.4%** |

**+7.0 pp**, and PoseBusters fell by **−7.0 pp**. Bond-length medians are
identical (1.438 A both); it is entirely the tail (out-of-window bonds 0.29% ->
0.69%), which per molecule compounds over ~24 bonds.

Fixed by separating the two questions: `clash_probe` decides *which positions*
to re-decide (in the wall), `accept_probe` decides *which replacement* is
allowed (out of the wall and still bonded). `make_clash_probe(..., bonds=...)`
builds the second. Arm rerunning.

## The constraint ladder

Each step adds one window to what a REPLACEMENT must satisfy; the selection of
which positions to re-decide never changes. Every window is read off the control
arm's own distribution, so none of them is a tuned constant.

| arm | vina (median) | vina (mean) | PB | strain | clash count | paired vs control |
|---|---|---|---|---|---|---|
| confidence (control) | −0.005 | +1.704 | **0.805** | **346** | 2.42 | — |
| clash, wall only | **−1.246** | **−0.636** | 0.718 | 503 | 1.06 | −1.215 |
| `+` bond length | −1.164 | −0.417 | 0.740 | 465 | 1.06 | −1.191 |
| **`+` angle and self-clash** | −1.156 | −0.239 | **0.759** | **432** | 1.18 | −1.161 |

The trade is explicit and small: the two geometry windows buy **+0.041
PoseBusters and −71 strain for 0.09 kcal** of median score. The per-check
breakdown moves the way each window aims:

| check (percent failing) | control | wall only | `+`bond | `+`geom |
|---|---|---|---|---|
| bond lengths | 6.2% | 13.1% | 8.1% | 7.9% |
| bond angles | 6.6% | 14.1% | 12.8% | **7.9%** |
| self-clash (1-4 and beyond) | 12.1% | 21.0% | 18.1% | **15.9%** |

None returns fully to the control, for two reasons that are in the design: the
windows are enforced on the atom being moved while the refiner then moves the
whole molecule, and a position whose every candidate is rejected keeps the
model's argmax rather than blocking.

## The two levers overlap

Clash-leaf pruning fires on **50%** of the deployed arm's molecules (0.99 atoms
each) but only **27%** of this arm's (0.54 each). Clash-ordered decoding has
already removed about half of what pruning would have taken — they are aimed at
the same atoms.

Measured, they are sub-additive by a factor of five but not redundant:

| arm | vina (median) | vina (mean) | PB | strain | heavy |
|---|---|---|---|---|---|
| confidence (control) | −0.005 | +1.704 | 0.805 | 346 | 22.30 |
| clash+geom | −1.156 | −0.239 | 0.759 | 432 | 22.28 |
| **clash+geom `+` pruning** | **−1.623** | **−1.271** | 0.767 | 418 | 21.66 |

Pruning on top of clash-ordered decoding: **paired −0.184 kcal**, p=3e-14, better
on 76% of targets — against **−0.954** when it was applied to the deployed arm.
Chemistry moves slightly the right way as well (PB 0.759 -> 0.767, strain 432 ->
418, QED 0.424 -> 0.438); the cost is 0.62 heavy atoms, taking the median
molecule to 21.66 against the reference's 22.

The marginal medians differ by 0.467 while the paired difference is 0.184. The
paired figure is the effect; the other is the difference of two medians and is
not one. Both are printed so the gap between them cannot be quoted as a result.

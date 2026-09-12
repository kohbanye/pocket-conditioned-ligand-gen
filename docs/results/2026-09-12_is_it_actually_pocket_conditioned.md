# Is the generation actually pocket-conditioned? The control, and the number

ProLIT is a **pocket-conditioned** ligand generator and this repository had no
control for that. The circumstantial evidence was not encouraging: giving the
model more pocket information (`pctx`) moved neither the score nor the clash rate
(paired 0.000, p=0.41), and the LM names the data's own code only 22.6% of the
time.

## The control

Condition on **someone else's pocket** while still decoding into **this
target's frame**. A derangement over the canonical 100 (each target prompted by
the next one alphabetically, so the pairs are unrelated proteins), same seed,
same weights, same everything else. `--prompt-receptor` / `--prompt-ref-ligand`
were added for this.

Taking the other pocket's *frame* as well would put the molecule somewhere
arbitrary in space and measure that instead. The question is whether the **codes
the model chooses** depend on the pocket, so only the prompt is swapped.

## The molecule is entirely a function of the pocket

56 targets paired, same seed, prompt the only difference:

| | |
|---|---|
| identical SMILES to the correct-pocket arm | **0.0%** |
| Tanimoto to the correct-pocket molecule | **0.076** |
| same heavy-atom count | **3%** |
| coordinate RMSD where the count matches | 6.95 A |

Not "somewhat different" — unrelated, down to the length.

## And it is conditioned usefully

Clean 79:

| arm | score | min | PB | strain | unique | heavy |
|---|---|---|---|---|---|---|
| correct pocket (best arm) | **−1.308** | **−4.759** | 0.515 | 870 | 0.691 | 22.07 |
| swapped pocket | +22.042 | −3.399 | 0.567 | 678 | 0.623 | 18.91 |

| question | answer |
|---|---|
| raw score | **+20.937** (p=6.8e-14, correct better on 96%) |
| after local optimisation | **+1.902** (p=6.4e-11, 85%) |
| **and after matching size** | **+1.003** (p=1.3e-04, 84%, 32 targets) |

The 21 kcal is a molecule built for another pocket colliding with this one, which
is not informative on its own. Local optimisation removes it. Of the 1.902 that
remains, about half is the wrong pocket asking for a **smaller** molecule
(18.91 heavy atoms against 22.07) — real conditioning, but size.

**Net: about 1.0 kcal of genuinely better binding that is neither placement nor
size.** That is the pocket conditioning, measured for the first time.

## Two things worth keeping

**PoseBusters and strain are BETTER for the wrong pocket** (0.567 against 0.515,
678 against 870), because smaller molecules are easier to get geometrically
right. A chemistry metric can improve while binding collapses by 21 kcal. They do
not track binding and must never stand in for it.

**The size-matched figure rests on 32 targets** — the subset where the two arms'
heavy-atom distributions overlap enough to compare inside a band. The unmatched
+1.902 is on all 79.

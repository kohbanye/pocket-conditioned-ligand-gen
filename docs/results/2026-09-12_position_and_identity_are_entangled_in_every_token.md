# The placement error is what sampling this representation costs

## Where it is born

`min_rmsd` — how far Vina's local optimiser has to move a pose — read at each
stage of the pipeline, clean 79:

| stage | displacement | kcal it recovers | score |
|---|---|---|---|
| **LM + decoder only** (`dfs_full_noref`) | **1.783 A** | 7.718 | +3.394 |
| + refiner + clash order + prune | 1.222 A | 3.199 | −1.308 |
| published stack, noref | 1.946 A | 9.219 | +5.562 |
| published + rigid projection | 1.510 A | 5.156 | +1.351 |
| **crystal ligand** (control) | **0.136 A** | **0.017** | −6.871 |

Paired, the whole repair stack moves the displacement 1.814 -> 1.238 A
(**−0.501**, p=9.7e-13). **The error is born in the language model plus the
decoder and everything downstream is repair, which recovers 28% of it.** The
tokenizer's own round trip moves a centroid 0.191 A — about a tenth.

## What in the code choice causes it

Take the reference ligand's own codes and replace a fraction with codes sampled
from the model itself, measuring against `decode(true codes)` so the tokenizer's
error cancels. **Teacher-forced** replacement conditions each draw on the true
prefix (per-token noise alone); **free-running** conditions on what was actually
emitted (noise plus accumulation). 79 clean targets, 3 repeats, temperature 0.7,
top-p 0.95.

| fraction replaced | teacher-forced (A) | free-running (A+B) | **B** |
|---|---|---|---|
| 0.10 | **0.359** | 0.365 | 0.006 |
| 0.25 | 0.585 | 0.618 | 0.033 |
| 0.50 | 0.845 | 1.134 | 0.289 |
| 0.75 | 1.137 | 1.472 | 0.335 |
| **1.00** | **1.448** | 1.344 | −0.105 |

At f=1.0, paired: **−0.144 A, p=0.183** — no difference.

**Accumulation is not the mechanism.** Per-token noise alone produces the whole
effect, and it produces it immediately: replacing **2 codes of 23** already
displaces the centroid **0.359 A**, nearly double the tokenizer's entire
round-trip centroid movement. Replacing all of them gives **1.448 A**, which
brackets the deployed generator's 1.783 A from its own optimum.

## The cause, stated

**Every token carries position and identity together.** The descriptor's `coord`
field is `(r, theta, sin phi, cos phi)` **relative to the pocket centroid** — so
an atom's absolute placement is baked into the code that names it. The LM's
distribution over codes has entropy 0.925 at 22.6% top-1, and one changed code
moves the *whole molecule*, not just its own atom (the same non-locality the
prefix-decode drift of 0.930 A showed from the other side).

So **there is no way to sample a different molecule without also moving it.**
The 1.78 A is not a bug, a calibration error, or a weak model — it is the price
of sampling this representation, and it is 66% of the reported gap.

This subsumes the session's other findings rather than competing with them: the
shape is right and the placement is not (84/16); the direction is
molecule-specific, not a global bias; the repair tools saturate because they
repair; and the molecules redock to within 0.447 kcal of crystal ligands.

## What would resolve it

Factor the representation so that **what** the molecule is and **where** it sits
are different variables:

* codes over **placement-invariant** geometry — internal coordinates, or
  coordinates in the molecule's own frame — so sampling chemistry cannot
  displace the molecule;
* an explicit **pose** (one SE(3) per molecule) predicted separately and trained
  directly against the crystal pose, which is a supervised target the corpus
  already has.

Both halves are then trainable against something real, and the 1.4 A that
sampling currently costs goes to zero by construction. This invalidates every
token stream and every checkpoint — it is a new tokenizer — which is why it is
recorded with the measurement that justifies it rather than attempted as a
tweak.

# The readout change transfers; the input repair did not

2026-09-14. Two development-set effects were measured on the same cohort, with
the same four seeds and the same control. Both were carried to CASF this
afternoon. They behaved oppositely, which is the useful part.

| | dev ΔR | CASF ΔR | CASF Δrho | n_better (R / rho) |
|---|---:|---:|---:|---|
| input repair (heavy crop + validated chemistry) | +0.0303 | −0.0015 | −0.0110 | 1/4 · 0/4 |
| M14 `balanced` readout | +0.0167 | +0.0075 | +0.0123 | 3/4 · 3/4 |
| M14 `allatoms` readout | +0.0171 | +0.0070 | +0.0171 | 3/4 · **4/4** |

Control is `e250_dev_ca_legacy`, paired seed by seed, all at 285-complex
coverage against GenScore's matched 0.8127 / 0.7281.

## Verdict

The pre-registered rule is `n_better`/4 on **both** metrics. Neither readout
reaches it (3/4 on R), so **neither is established** by the rule the round was
run under. But the contrast with the input repair is real and worth stating:
the readouts move both metrics in the right direction on nearly every seed,
and `allatoms` improves ranking rho on **all four**, which is the metric the
model is furthest behind on.

`allatoms` seed 7 reaches rho 0.7281 -- equal to GenScore to four decimals. It
is one seed of four and its R is 0.0223 short, so it is a tie on one metric in
one draw, not a win.

## Where it leaves the standings

| family (4 seeds) | R | rho |
|---|---:|---:|
| GenScore (285) | **0.8127** | **0.7281** |
| `e250_dev_allatoms` | 0.7840 (−0.0286) | 0.6807 (−0.0474) |
| `e250_dev_balanced` | 0.7845 (−0.0281) | 0.6759 (−0.0522) |
| `e250_dev_ca_legacy` (control) | 0.7770 (−0.0356) | 0.6636 (−0.0645) |

**No arm beats GenScore on both metrics, at any seed.**

## Why this one transferred and the other did not

Not established, but the shapes differ in a way worth recording. The input
repair changed what the encoder *sees* -- a different pocket and a different
ligand chemistry than the VQ and MLM were trained on, so its dev gain may have
been the head adapting to a shifted input distribution it could only exploit on
that distribution. The readout changes only which encoder states the head
*averages*, with the inputs, the tokenizer, the corpus and the parameter count
all fixed. A benefit there is a property of the representation, which is the
kind of thing that travels.

That the protein rows carry usable affinity signal at all is the finding: the
current readout pools ligand rows only, on the argument that pocket rows are
identical across the poses of one target. That argument holds for pose
rescoring, where it came from. It does not hold for affinity, where each
complex has its own pocket -- and averaging those rows in is worth +0.017 rho.

## What is untested

`balanced` and `allatoms` heads were trained on `ca_legacy`, the development
cohort -- which is the *worse* corpus: its own control sits below
`e250_union_noconf` (R 0.7971) and below `e250_rot1` (rho 0.6953). The readout
has never been combined with the corpus that leads on R. That combination is
the obvious next test, with the caveat that the two effects must not be assumed
additive; H7 and H9 both showed corpus effects failing to stack.

---

## CORRECTION (same day, evening)

The section "Why this one transferred and the other did not" is **withdrawn**.

The readout was retrained on `union_noconf`, the corpus that leads on scoring R,
changing nothing but `--readout`. Against that corpus's own control it scores
**0/3 seeds on rho, mean −0.0515** (`allatoms`) and **0/3, mean −0.0561**
(`balanced`) -- the opposite sign to the +0.0171 measured here on `ca_legacy`,
with the same tokenizer, the same MLM and the same pooling arithmetic.

So the argument that a readout benefit is "a property of the representation,
which is the kind of thing that travels" is false as stated, and the conclusion
that "the protein rows carry usable affinity signal" does not hold beyond the
cohort it was measured on. Details: `2026-09-14_the_readout_gain_is_corpus_specific.md`.

The table of measurements above stands; only the interpretation was wrong.

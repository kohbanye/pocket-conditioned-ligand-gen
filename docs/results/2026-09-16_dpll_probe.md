# The pocket-ablation probe measures nothing the old `pll` column did not (2026-09-16)

Free: no training, one GPU, 75 s for all 285 CASF core complexes.
`experiments/affinity_loop_20260914/dpll_probe.py` -> `dpll/dpll_casf.csv`,
judged by `dpll_verdict.py` -> `dpll/verdict.txt`.

## What was measured

For each complex the crystal pose is encoded once (frame 0, no rotation, the
same path `_score_complex` uses) and the ligand span is scored twice with the
same MLM:

    pll_ctx   = PLL(ligand | pocket tokens)
    pll_nopkt = PLL(ligand | every pocket codebook token replaced by <mask>)
    dpll      = pll_ctx - pll_nopkt

`<mask>` rather than deletion, so the length, the markers and every position
index stay identical and the ablated input stays inside the distribution the
encoder was pretrained on.

**Wiring check**: `pll_ctx` reproduces the `pll` column of
`e250_unionnc_corr_v3_s58_f32`, corr 0.99994, **median |diff| exactly 0**,
257/285 below 1e-3, max 0.158 on 2wnc. The residue is TF32 nondeterminism across
GPUs, not a different sequence.

## Result: the ablation carries nothing

| predictor | rho(per-target mean, per-target bias) | honest LOO R | null p |
|---|---:|---:|---:|
| `dpll` | +0.2754 | 0.8229 -> **0.8262** | 0.005 |
| `pll_ctx` (already in every dump) | +0.3129 | 0.8229 -> **0.8261** | 0.005 |
| `pll_nopkt` | -0.0086 | 0.8229 -> 0.8195 | 0.625 |

**`dpll` and `pll_ctx` are worth the same thing, and `pll_nopkt` is worth
nothing.** The pocket-ablated term has no target-level signal at all, so the
difference is `pll_ctx` minus a constant-ish quantity: masking the pocket costs
every complex about the same 9.3 nats/atom (sd 0.54 against `pll_ctx`'s 1.49).

-> **The new measurement is refuted.** Anything it can do was available from a
column that has been written into every dump since the bench existed.

## What the surviving number is, and is not

Used as a **per-target offset predictor** -- regress per-target bias on the
per-target mean, subtract from every complex of that target -- `pll` is worth
**+0.0032 R**, leave-one-cluster-out, and beats a label-preserving permutation
null at **p = 0.005** (null 0.8199 +- 0.0014). That makes it the first
label-free covariate in this campaign to survive a nested test; the 18-covariate
offset predictor gave rho 0.088 at p = 0.527.

It is still not enough and does not change any conclusion:

* **0 of 200 null draws reached 0.83, and neither does the honest value** (0.8262
  against the 0.8229 baseline; R > 0.83 needs +0.007).
* Used the way it was tried before -- blended per complex -- it is worth
  +0.0007 in-sample and **-0.0011 leave-one-cluster-out**, i.e. nothing, which
  reproduces the earlier refutation rather than overturning it.
* **The target-level correlation was already on record.** The overnight session
  measured target-level corr(pll, bias) = -0.338 (opposite sign convention, same
  magnitude) and called it "about 2.5 SE on 57 points, do not ride it". The only
  new thing here is the nested-LOO and permutation-null evaluation of it, which
  says it is real and tiny.
* **It is the correction least defensible under the open leak question.** `pll`
  is the MLM's familiarity with the complex, and `vq_e250_lig3`'s CASF membership
  cannot be verified from the checkpoint. Familiarity does not predict accuracy
  (corr(pll, |residual|) = -0.056), which is why the leak disclosure stands, but
  a headline that *corrects predictions using familiarity* would have to argue
  that separately. Not worth +0.003.

## Standing

Not adopted. The reported configuration stays the pre-registered 36-model mix.
Levers refuted: the pocket-ablation probe (new), and the per-complex blend of
`pll` (re-confirmed).

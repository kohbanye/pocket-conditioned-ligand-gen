# PDBbind IC50 in the union: refuted, and it costs R as well as rho

2026-09-14 23:57 JST. Pre-registered in
`docs/notes/2026-09-14_prereg_union_ic50.md` before either corpus was built;
applied mechanically by `experiments/affinity_loop_20260914/union_ic50_verdict.py`
(written while the corpora were still tokenizing). Raw output in
`union_ic50_verdict_output.txt`.

Before the verdict ran: all 28 chain jobs verified (`qacct` exit 0, no log
errors, checkpoints present, 285/285 non-NaN predictions in every dump), both
corpora checked against their builders' own counts, and both new PDBbind halves
leak-audited.

## Verdict: DOES NOT WORK (D1 False, D2 False)

The two arms differ in exactly one flag, `--affinity-types`, on both halves.

| f32 family ensemble | R | from between | from within | rho |
|---|---:|---:|---:|---:|
| treatment `e250_unionic50` (24,895 train docs) | 0.8029 | **0.3611** | 0.4417 | 0.6737 |
| control `e250_unionkdki` (14,372 train docs) | 0.8083 | **0.3719** | 0.4364 | 0.6965 |

* family dR **-0.0055**, paired 2/4 seeds (s7 +0.0053, s11 -0.0264, s12 -0.0225, s13 +0.0052)
* family drho **-0.0228**, paired 1/4 seeds (s7 -0.0281, s11 -0.0684, s12 -0.0105, s13 +0.0070)
* f16 agrees: dR -0.0053 (2/4), drho -0.0298 (1/4)

D1 wanted dR >= +0.010 on >= 3/4 seeds; D2 wanted drho >= -0.010 on >= 2/4.
Both fail.

**+10,523 PDBbind complexes (+73% of the corpus) made the model worse.** The
hypothesis was that PDBbind-curated IC50 might not carry BioLiP IC50's ranking
cost, because the union's historical +0.035 R was a cross-source effect. It
carries the cost AND loses R, which BioLiP IC50 did not (that was R +0.005,
rho -0.018). Thirteen levers are now refuted or neutral.

**The between-target part fell** (0.3719 -> 0.3611) while the within-target part
rose slightly. The deficit against GIGN (0.4128) and OnionNet-2 (0.4290) is
between-target, so the one lever that moved corpus size at scale moved the weak
half the wrong way.

## Headline (computed regardless, as pre-registered)

f32 z-sum of the 4 treatment arms, twin set 22 (Amendment 2's 12, plus the
`union_noconf` audit's 7, plus `1q8u 1ydt 3ejr` from the new IC50 half):

| | R | rho |
|---|---:|---:|
| all 285 | 0.8029 | 0.6737 |
| clean 263 | 0.8090 | 0.7286 |
| GenScore (all 285) | 0.8127 | 0.7281 |

FAIL: R -0.0098 against GenScore, -0.0271 against the 0.83 bar, rho -0.0544.

## What did replicate: the union itself

The control is a fresh BioLiP+PDBbind union, built today from the recovered
PDBbind tree with different exclusion lists, a different val split
(`--max-heavy 60 --seed 0` rather than 80 / 7), and 14,372 train docs against
`union_noconf`'s 15,351. It lands at **R 0.8083 / rho 0.6965** against
`union_noconf`'s **0.8101 / 0.6930** -- within seed noise of the corpus it was
meant to reproduce. The union recipe is reproducible from raw data; it is the
IC50 extension that fails.

## Leak status of these arms

Both new PDBbind halves were audited under the frozen policy before this was
reported. `casf_pdbids_in_candidates` is **empty in both**, so no CASF core
complex is itself a training candidate. Twins: 13 (kdki half, none new), 16
(ic50 half, 3 new: `1q8u 1ydt 3ejr`, including the only exact-criterion twins
found anywhere -- `1q8u` via 2gnl, `1ydt` via 3vqh). Reconstructions matched the
build logs exactly for both. As established with `union_noconf`, the twin screen
is a **lower bound** (CASF complexes self-flag only 190/285 from their own
PDBbind files), so the clean column is a sensitivity analysis, not a
certification.

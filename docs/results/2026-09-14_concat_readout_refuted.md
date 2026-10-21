# The concat readout does not recover the between-target signal (2026-09-14 21:56)

Pre-registered in `docs/notes/2026-09-14_prereg_concat_readout.md` before any
concat arm was implemented; applied mechanically by
`experiments/affinity_loop_20260914/concat_verdict.py`; raw output saved as
`concat_verdict_output.txt` beside it. Before running it: all 8 heads verified
(checkpoints, clean log end, `qacct` failed 0 / exit 0) and all 16 inference
jobs verified (`qacct` exit 0, no log errors, 285/285 non-NaN predictions).

## Verdict: DOES NOT WORK

P1 (between >= 0.395) **False on both corpora**. P2 (within >= 0.420) True on
both. P3 (R above control on >= 3/4 paired seeds) **False on both**.

### f32 (pre-registered basis), family ensembles

| corpus | arm | R | from between | from within | rho |
|---|---|---:|---:|---:|---:|
| union_noconf | concat | 0.7942 | 0.3570 | 0.4372 | 0.6737 |
| union_noconf | control (`ligand`) | 0.8101 | 0.3801 | 0.4301 | 0.6930 |
| kdki_ctl | concat | 0.7869 | 0.3480 | 0.4389 | 0.6860 |
| kdki_ctl | control (`ligand`) | 0.7908 | 0.3494 | 0.4413 | 0.7509 |

Paired dR by seed (same driver, so seeds do pair trajectories):
union_noconf s7 -0.0644, s11 +0.0123, s12 -0.0115, s13 -0.0277 (1/4);
kdki_ctl s7 +0.0234, s11 -0.0004, s12 +0.0082, s13 -0.0598 (2/4).

### f16 (reported beside)

| corpus | concat R / between / within / rho | control R / between / within / rho | better |
|---|---|---|---|
| union_noconf | 0.7962 / 0.3592 / 0.4370 / 0.6807 | 0.8107 / 0.3820 / 0.4287 / 0.6737 | 1/4 |
| kdki_ctl | 0.7894 / 0.3488 / 0.4405 / 0.6860 | 0.7900 / 0.3493 / 0.4407 / 0.7421 | 3/4 |

## What the prediction got wrong

The diagnosis argued that the pocket signal behind `allatoms`' higher
between-target part (0.4047 vs `ligand` 0.3801) is present in the
representation and the `ligand` readout throws it away, so handing the head the
pocket mean separately would keep it without diluting the ligand side.

The ligand side was indeed not diluted (P2 holds; within-target parts within
+/-0.007 of control). But with the pocket mean available, the between-target
part **fell** on union_noconf (-0.023) and stayed flat on kdki_ctl (-0.001). The
head did not turn pocket features into transferable target-level affinity.
Ranking also fell at f32 on both corpora (-0.019, -0.065).

Readings, neither tested:

* **Most economical:** the 0.025 between-target gap between `allatoms` and
  `ligand` was measured on ONE family of 4 seeds, where per-seed R has sd
  ~0.05. It was probably seed noise, and the diagnosis built a mechanism on it.
  This is the same one-family error that sank f32; the pre-registration's
  two-corpus requirement is what caught it.
* **Alternative:** a separate pocket vector lets the head fit target identity
  on training targets, which does not transfer to CASF targets.

## Consequences

* Readout choice is not a lever. Twelve levers are now refuted or neutral:
  interface contacts, pocket context, within-cluster compression, congeneric
  difficulty, metal coordination, zero-shot `pll`, cross-generation ensembling,
  MLM continuation, frame count, IC50 labels, the head-recipe lead (noise), and
  the concat readout.
* The concat headline (`2026-09-14_prereg_concat_headline.md`) is still computed
  as pre-registered (rule 4: regardless of this verdict) once the `union_noconf`
  leak audit lands. Its number will be recorded there, not predicted here.

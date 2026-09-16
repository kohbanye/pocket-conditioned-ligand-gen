# Pre-registered headline for the concat readout (2026-09-14 ~22:00)

Written BEFORE `concat_verdict.py` was run and before any CASF metric for any
concat-readout arm was computed or read. At the time of writing, 8 of the 16
concat dumps exist on disk (all `e250_unionnc_concat`, both frame counts); no
metric has been computed from them and none has been looked at. The other 8
(`e250_kdkictl_concat`) are still being produced.

This is separate from the lever pre-registration
(`2026-09-14_prereg_concat_readout.md`), which asks whether the readout works
(P1-P3, relative to controls). This document fixes, in advance, what counts as a
headline built from it. The earlier headline
(`docs/results/2026-09-14_prereg_headline_verdict.md`) failed and stands
unchanged whatever happens here.

Applied mechanically by `experiments/affinity_loop_20260914/headline_concat_prereg.py`.

## Rules

1. **H1 membership (the headline).** Exactly the 8 concat arms:
   `e250_unionnc_concat{,_s11,_s12,_s13}` and `e250_kdkictl_concat{,_s11,_s12,_s13}`.
   Frozen in `headline_concat_membership.json`, fingerprint
   `afa665c4a835316dcd97bef6749e28d5b41ca35a2a703d3d2781f4b8d7f6cde7`
   (captured with `variants.py` sha256 `abc2450b...`). A change to any of these
   arm definitions makes the script refuse.
2. **Basis.** f32, z-sum ensemble (identical `zsum` to the earlier headline), all
   285 CASF-2016 core complexes. f16 is reported beside it and cannot rescue a
   failed f32 verdict.
3. **Goal.** R > 0.8127 AND rho > 0.7281 (GenScore, measured here) AND R > 0.83.
   All three, same as the failed headline.
4. **Computed regardless of the lever verdict.** Gating the headline on P1-P3
   would let the lever result decide whether a headline exists at all.
5. **H2, a pre-declared secondary (not the headline).** z-sum of the 8 concat
   arms plus the 65 arms frozen in `headline_prereg_membership.json`. If H2
   passes and H1 fails, it is reported as exactly that -- a secondary comparison
   passing -- and not as the headline. No other pool may be reported as a
   comparison against the goal.
6. **Single seed.** The best concat seed by R is reported and labelled as a
   maximum over 8 seeds chosen on the test set. It claims nothing.
7. **Leak sensitivity.** clean = 285 minus T, where T = the 12-complex set of
   Amendment 2 UNION whatever the `union_noconf` audit finds.
   * `union_noconf` = `e250aff_r1` (BioLiP) + `e250aff_pdbbind_noconf` (PDBbind
     v2020, KD/KI). **It was never screened on its own** (found 21:50): the
     existing audits cover the dev cohort, `kdki_ctl` and `ic50`, and neither
     half of `union_noconf` has a site manifest. The earlier headline's clean
     column therefore rests on an incomplete twin set for its `union_noconf`
     families; that is disclosed here and will be quantified when the audit
     lands.
   * The audit reads no affinity labels and no predictions, uses the frozen
     policy (exact chain sequence + ligand identity; Morgan r2/2048 Tanimoto
     >= 0.9 with sequence identity and coverage >= 0.9), and its result is
     applied whatever it is. The script refuses until
     `leak_audit_general/out_union_noconf/TWINS.json` exists.
   * **If any CASF core pdbid itself is among the training candidates**, that is
     a leak, not a sensitivity item: the script reports INVALID and computes
     nothing.
8. **Refusals.** Any missing dump (either frame count, H1 or H2 members), a
   concat fingerprint mismatch, a prior-membership fingerprint mismatch, or a
   missing/malformed union audit.

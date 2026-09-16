# The input repair could not be measured on CASF at all

2026-09-14. Picks up the affinity push from the 2026-09-13 development round
(`experiments/affinity_loop_20260914/RESULT_JA.md`), which found two input
defects by reading structures rather than residuals, repaired both, and measured
+0.030 development Pearson for the pair -- and then could not carry that to
CASF, because **the evaluation path had no way to supply one of the two repaired
inputs**. This note records the gap, the fix, and what is now being measured.

## The two defects, restated

| defect | evidence | development effect |
|---|---|---|
| ligand formal charge never reaches the descriptor | the molecule is rebuilt from elements and bond orders, which cannot carry charge | ΔR +0.003 alone |
| pocket residues selected by CA distance | 124 of 283 CASF complexes lose a standard residue lying within 4 A of the ligand; the 50-residue cap causes none of it | ΔR +0.008 alone |
| both | -- | **ΔR +0.030, all four seeds** |

## Why CASF could not be measured

`--pocket-distance heavy` was wired into the benchmark. Validated chemistry was
not: `preserve_chemistry` exists only on `parse_ligand_pdb_text`, the BioLiP
route, which recovers chemistry from a **CCD template**. CASF ligands arrive as
SDF and have no CCD template in that path, so the arm's head would have been
scored on rebuilt ligands -- different inputs than it was trained on, with
nothing in a dump to reveal it.

CASF ships a second copy of every core ligand as a crystal MOL2, and that is the
available chemistry:

| check | count |
|---|---:|
| MOL2 sanitizes | 285 / 285 |
| SDF sanitizes | 193 / 285 |
| MOL2 heavy atoms match the SDF in order and to 0.01 A | 282 / 285 |
| accepted by the descriptor's `validated_heavy_mol` contract | **282 / 285** |
| charged ligand atoms recovered, validated vs rebuilt | **257 vs 14** |

Probe: `validated_mol_from_mol2_text` over the whole core set, run before the
production path was written. The three rejects (3ary, 3bv9, 4eky) are the known
SDF/MOL2 coordinate disagreements -- 3ary and 4eky differ by up to 1.74 and
1.30 A, and 3bv9's SDF does not sanitize at all. They fail closed to the legacy
route: scoring a different pose is a worse error than a missing charge, and full
285 coverage is required, so unlike training they cannot simply be dropped.

## What changed

- `prolit.api.validated_mol_from_mol2_text` builds a `validated_v1` payload from
  a sanitizable MOL2. Atoms **and** bonds come from the same molecule that
  produced the payload, so its geometry and chemistry cannot disagree with each
  other; it verifies them against the SDF the bench scores, re-runs the
  descriptor's own acceptance check, and returns `None` rather than failing
  after encoding has started.
- The input condition now lives on the **arm** (`Variant.pocket_distance`,
  `Variant.chemistry`), not only on the CLI. A head trained on repaired inputs
  and scored on rebuilt ones is a different model, and the registry is where
  what-an-arm-means is defined. `--pocket-distance` / `--chemistry` still
  override, for deliberately measuring an arm under inputs it never saw.
- Each dump gets a `<head>.chemistry.json` sidecar naming every complex that
  fell back, so a partially-fallen-back run cannot read as a validated one.

## What is being measured now

16 arms registered (`e250_dev_{ca,heavy}_{legacy,validated}` x seeds
7/11/12/13); CASF submitted for the two that answer the transfer question,
paired by seed, 16 frames, against the round's own control `e250_dev_ca_legacy`
-- not against `e250_rot1`, which would credit an input repair with a corpus
change as well.

**These CASF numbers will be leak-flagged.** 20 of the 7,714 training sites
share an exact protein sequence AND ligand with a CASF core complex. The arms
carry that in their `notes`. The cleaned 7,666-site cohort
(`goal_verification/eligible_cache_inputs/cohort.json`, 43 source exclusions
plus 5 native-ineligible) exists precisely so the headline number can be
retrained without them; this run only decides whether that retrain is worth it.

Development absolute values (R ~0.58-0.61 on 492 documents) are **not**
comparable to CASF values, and the +0.030 input effect and the +0.017 readout
effect (M14) are not additive.

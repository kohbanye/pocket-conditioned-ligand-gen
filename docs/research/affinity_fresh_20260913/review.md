# Final critical/editorial/ethics review — 2026-09-13

**Verdict: PASS for an exploratory research report; minor precision corrections recommended.** This is not certification that the existing models or datasets are leak-free, nor endorsement of a claim of superiority. No such claims are made by the reviewed report.

Reviewed `REPORT_JA.md`, `analyze.py`, the current chemistry summary, selected-pair table, structure contact table, execution-contract audit and previously inspected benchmark code/baseline failure logs. This checkpoint did not independently re-open every external paper; source-specific verification remains the literature review's responsibility. It did not train or rerun a model. Final freshly regenerated artifacts should be checked against the report's rounded figures before delivery.

## Checks that pass

- Main metrics use matched 283 IDs for Pearson and 55 complete five-ligand groups for macro Spearman, with explicit disclosure that this is not full CASF. The analysis asserts unique prediction IDs and matched labels/clusters. The two real GenScore failures and their graph-stage errors are correctly described.
- Report separates individual seeds, average **metrics** and average **predictions**. The latter is correctly labeled a three-model ensemble and is not used to claim superiority to a single GenScore checkpoint.
- Cluster bootstrap samples shared target blocks with replacement and uses identical sampled indices for both methods. Report correctly says intervals condition on already-trained predictions and exclude retraining uncertainty. Failure to reject a gap is not misrepresented as equivalence.
- The report explicitly distinguishes observed input/representation defects, their association with residuals, and still-unproven causal affinity improvement. No promised gain or SOTA claim is made.
- Raw-SDF charge disagreement 156/285 is no longer described as proven corruption of valid source chemistry. Sanitizable-SDF and mapped-sanitized-MOL2 comparisons are separately counted. The report warns that sanitizability does not establish assay-state protonation truth and rejects blind copying of invalid charge records.
- Geometry distances use exact inference SDF heavy coordinates; sanitized MOL2 supplies chemical descriptors/fingerprints. This avoids treating invalid raw SDF charge states as chemical ground truth. Coordinate-set distance is an appropriate additional check, though it is not a chemical atom mapping by itself; atom mapping is separately addressed by the chemistry audit.
- Missing contact side chains are supported by actual measured distances; hydrogen bonds/energetic significance are not asserted from distance alone. The report correctly notes that shared missing Mg cannot by itself explain a between-ligand affinity difference.
- Selected wrong pairs are explicitly post hoc, require consistent failure across three seeds and a sizeable measured difference, and are presented as mechanism hypotheses. They are not causal tests or unbiased performance strata.
- Chemistry preprocessing improvements and extra training data are not automatically credited to tokenizer novelty. Main proposed head is fixed; preprocessing, tokenizer and LM changes have separate ablations. The 2×2 tokenizer/LM plan acknowledges that different code spaces require LM retraining.
- Exact-ID exclusion is not equated with target/scaffold novelty. All-stage leakage provenance remains unresolved, accurately described. CASF-directed investigation is labeled exploratory, with a fresh locked confirmatory set required.
- Boltz versus crystal-input comparisons, released versus retrained GenScore, and single versus ensemble literature numbers are distinguished.
- Research-only scope and lack of new training are disclosed. No production benchmark or external artifacts were changed by this audit.

## Minor corrections / final consistency checks

1. `analyze.py` bootstrap comment says “preserves all five ligands,” but its Pearson bootstrap contains 55 blocks of five and two blocks of four. Replace with “preserves observed target blocks.” The computation and report are correct; the comment is imprecise.
2. `analyze.py` selects six target-distinct failure pairs; the report table displays five. State that these are the first five ranked cases and that the sixth is included in `selected_cases.csv`/the ligand grid, or display all six. The report currently explains the selection rule but not this display cutoff. The five are indeed the first five selected clusters, although GluK1 and glycogen rows are reordered for narrative clarity.
3. Confirm after the final analysis rerun that all rounded metrics, group-bias figures and selected-case counts still match artifacts. Chemical stratification may change under the revised valid-MOL2 route even though prediction metrics do not.
4. Preserve the narrow meaning of “input chemistry inconsistency”: the audit establishes fields lost/disagreed, not that every charge reference is physiologically correct. Current report does so.

## Nonblocking limits to retain in downstream work

- Source snapshots and prediction hashes are partially captured; full corpus/structure/checkpoint/exclusion lineage is not reconstructed. Do not convert this report into an audit certificate.
- The 57 CASF groups may not constitute 57 biologically independent target families. Target-block bootstrap improves on independent-complex sampling but does not establish independence among homologous groups; a family-aware sensitivity analysis would strengthen a confirmatory study.
- No significance test is used to certify feature-defined subgroup differences; this is appropriate for the exploratory multiple-stratum inspection. Do not later attach causal language to the same group table.
- A fixed simple probe gives evidence of decodable information, not a proof that information is absent when a finite-capacity probe fails. The report's probe-capacity caveat is important and should remain.


Final resolution (root): bootstrap comment and six-selected/five-displayed explanation corrected. Final metrics and 283/55-cluster/550-pair invariants checked. Added all-heavy-atom pocket omission counts (124 complexes/181 residues) and SDF/MOL2 coordinate discrepancies (3ary, 4eky), explicitly qualifying the 4eky mechanistic case. Source syntax/local links validated.

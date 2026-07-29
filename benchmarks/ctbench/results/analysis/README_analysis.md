# Evidence-based analysis: separate_4096 vs joint_nocasf vs DiffSBDD

CrossDocked2020 100-pocket test set, 100 samples/pocket. CPU analysis of already-generated
structures (no GPU / no docking re-run). RDKit via sbdd-bench venv.

Sources: `results/generation/{joint_nocasf,separate_4096}/per_molecule.parquet`,
`sbdd-bench/results/per_molecule.parquet` (diffsbdd), raw SDFs under `sbdd-bench/outputs/`.

## Files here
- `composition_by_model.csv` — mean size/ring/element descriptors (RDKit over largest fragment).
- `ring_features_by_model.csv` — disconnected frac, 3/4-ring/macrocycle frac, no-aromatic frac.
- `ring_size_counts_by_model.csv` — ring-size histogram.
- `per_target_{joint,separate,diffsbdd}.csv` — per-target aggregates (100 targets each).
- `posebusters_subchecks_sample.csv` — PoseBusters per-check pass rates (12 targets/model sample).

## Headline numbers (per_model)
| metric | joint | separate | diffsbdd |
|---|---|---|---|
| validity (sanitize) | 0.975 | 0.972 | 1.000 |
| connected (1 fragment) | 0.677 | 0.808 | 1.000 |
| pb_valid | 0.426 | 0.452 | 0.543 |
| clash_free | 0.630 | 0.626 | 0.718 |
| QED | 0.374 | 0.372 | 0.489 |
| SA | 5.55 | 5.53 | 4.52 |
| vina_dock (mean) | -6.36 | -6.80 | -7.35 |
| vina_dock median | -6.22 | -6.73 | -7.35 |
| frac vina_dock < -8 | 0.125 | 0.197 | 0.277 |
| heavy atoms (largest frag) | 15.6 | 17.6 | 20.9 |
| aromatic rings / mol | 0.09 | 0.09 | 0.46 |
| frac with NO aromatic ring | 0.91 | 0.91 | 0.63 |
| hit-rate | 0.019 | 0.050 | 0.079 |

## Q1 — why separate_4096 slightly beats joint_nocasf on generation
Driver = **connectedness + molecule completeness**, NOT per-atom chemistry.
- connected: 0.808 vs 0.677; separate wins on **82/100 targets** (Wilcoxon p<1e-4).
- vina_dock: separate wins **78/100 targets** (p<1e-4); shift is whole-distribution (median -6.73 vs -6.22) AND tails (frac<-8 0.197 vs 0.125).
- heavy-atom count of the docked largest fragment: 17.6 vs 15.6, separate wins 77/100 targets.
- own-bond SDF sanitize rate (12-target sample): separate 0.76 vs joint 0.60 (rest rescued by OpenBabel).
- **TIED** (all n.s.): QED (0.372 vs 0.374), SA (5.53 vs 5.55), aromatic rings, strain (median ~1.07e3 vs ~0.99e3), clash-free.
- Mechanism confirmed: within each model connected mols dock better than disconnected, and vina_dock correlates with size (r≈-0.8). Dedicated 4096 ligand codebook → less fragmentation + larger consistent single-bond ligand graphs → better vina/PB. It does NOT make atoms more drug-like.

## Q2 — why ours underperforms DiffSBDD
Gap is BOTH chemistry and 3D geometry, near-universal across targets.
- **No bond orders.** Our SDFs are 100% single bonds (0% double/triple/aromatic); DiffSBDD emits 73% single / 10% double / 17% aromatic. Consequence: 91% of our mols have zero aromatic rings vs 63% for DiffSBDD; ours win on aromatic-ring count on only 4-5/100 targets. This depresses QED and inflates SA.
- **Validity mechanism (B):** own bonds fail sanitize by VALENCE (single-bond over-saturation) 40% (joint) / 24% (separate) of the time; DiffSBDD fails only ~2.5% (kekulization). OpenBabel re-perception rescues ours to ~97% valid; ~2-3% stay invalid. DiffSBDD is 100% valid because it emits kekulizable bond orders.
- **Pose geometry (C):** PoseBusters localizes the loss to bond_lengths (joint 0.50 / sep 0.63 / diff 0.83) and bond_angles (0.61 / 0.77 / 0.82) and internal clash (0.86 / 0.90 / 0.96). Aromatic-ring-flatness is a trivial tie (ours have ~no aromatic rings). Strain median ~960 (ours) vs 357 (DiffSBDD).
- ours-vs-diffsbdd per-target win rates: vina_dock 16-27%, QED 8-10%, SA 17-19%, arom rings 4-5%, strain 9%, hit 2-9%; clash-free 40-42% is the closest.

## Key caveat
The aromaticity/QED/SA gap is largely a **representational limitation** of the atoms+single-bond VQ pipeline (no bond-order channel; unsaturation must be recovered post-hoc from imperfect 3D coords), not evidence the model "draws bad 2D molecules." The bond_length/bond_angle/strain gap is a genuine 3D-geometry quality gap of decode+refine vs E(3)-equivariant diffusion.

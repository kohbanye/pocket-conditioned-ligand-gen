# SBDD-bench comparison (3 targets: 2ity, 1iep, 3pbl; ~100 mols/target)

Vina lower=better; refiner = single-shot e07 on the raw generated poses.
Vina Dock unchanged by refinement (same molecule).

| Method | Validity | PB-valid | Clash-free | Vina Score | Vina Min | Vina Dock | QED | SA |
|---|---|---|---|---|---|---|---|---|
| DiffGui | 1.00 | 0.70 | 0.50 | -6.54 | -8.63 | -9.96 | 0.50 | 4.21 |
| TargetDiff | 1.00 | 0.50 | 0.60 | -4.76 | -6.68 | -9.00 | 0.37 | 5.16 |
| DiffSBDD | 1.00 | 0.49 | 0.70 | -4.40 | -6.48 | -8.40 | 0.52 | 4.73 |
| Ours (raw) | 0.96 | 0.21 | 0.30 | +1.21 | -6.45 | -9.37 | 0.39 | 5.74 |
| Ours+refiner | 0.95 | 0.21 | 0.56 | -4.03 | -7.34 | -9.37 | 0.40 | 5.83 |

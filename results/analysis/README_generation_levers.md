# Generation levers, measured (CrossDocked2020 100-pocket set, 2026-07-28)

Which pipeline change moves which metric, with effect sizes. All numbers are from
`results/generation/<arm>/per_molecule.parquet`; paired per-target tests via
`scripts/compare_arms.py`.

## Metric mechanics (learn this first)

| metric | what it responds to |
|---|---|
| `vina_dock` | the molecular **graph** only. `sbddbench/docking.py` writes a bare XYZ and runs `obabel -r`, so bond orders are re-perceived from coordinates and only the largest fragment is docked. Our SDF bond block and our refined coordinates are invisible to it (verified: writing real bond orders changed the mean by 0.001). Dominated by heavy-atom count, r ≈ -0.83 within an arm. |
| `vina_score` | the **pose as generated** — the metric the pose refiner actually drives. |
| `vina_min` | Vina's own local optimisation of that pose; the `score`→`min` gap is the recoverable local slack. |

The evaluator only re-perceives bonds with Open Babel when our own SDF *fails* to
sanitize (`sbddbench/molio.py`). Our writer emitted a single-bond-only bond block that
usually sanitized, so ~82% of molecules were scored aromatic-free.

## Arms

| arm | protocol |
|---|---|
| `separate_4096` / `joint_nocasf` | original 100 samples/target |
| `*_bo` | + Open Babel bond-order perception at write time |
| `*_cs` | + single-fragment + size floor 0.8x reference, first 100 acceptances from a 400-sample pool |
| `*_cs800` | `_cs` from an 800-sample pool |
| `*_fin` | 1150-sample pool, floor `max(0.8x ref, 20)` |
| `*_rx105` | `_cs` + pocket-aware clash relief |

## Effect sizes

### vina_dock (all 100 targets, DiffSBDD = -7.351)

| arm | vina_dock | heavy atoms | p vs DiffSBDD |
|---|---|---|---|
| separate_4096 | -6.804 | 19.2 | loses, 8e-05 |
| sep4096_cs | **-7.511** | 20.99 (matched) | 0.234 |
| sep4096_cs800 | **-7.803** | 22.44 | **0.00040** |
| joint_fin | **-7.661** | 22.44 | **0.0070** |

Bond-order perception alone moved `vina_dock` by 0.001 but fixed validity 0.97→1.00,
connectivity 0.81→1.00, QED 0.37→0.49 (parity with DiffSBDD). The `vina_dock` gain is
entirely the size/connectivity of the docked fragment. At matched size (`_cs`) ligand
efficiency ties DiffSBDD; at the larger sizes it does not, so report both.

DiffSBDD does **not** track the reference ligand size (corr 0.62); it regresses to its
own ~20.9-atom mean and runs 1.48x the reference on the smallest-reference third.

### vina_score (as-is)

| lever | effect | note |
|---|---|---|
| **pocket-aware clash relief** | **-2.922 → -5.593** (all 100) | `scripts/relax_in_pocket.py`, `--contact-scale 1.05 --w-pkt 10 --w-tether 0.5 --w-uff 0` |
| larger molecules + relief | -6.143 → -6.302 (9 tgt) | relief flips size from a liability to an asset |
| constrained sampling | +0.15 | |
| refiner swap `geo_v1` | -0.18 | worse |
| refiner swap `place_v2` | -0.89 | worse |
| temperature 0.85 | -0.07 | no effect |
| **vacuum UFF relaxation** | **-3.20** | harmful — see below |

The relief objective is a one-sided harmonic overlap penalty against a rigid pocket
plus a weak positional tether, with **no intramolecular force field**. Vacuum UFF fixes
strain (UFF energy 902 → 67) but expands the ligand out of complementarity; an
attractive 8-4 well is worse still, maximising contact rather than relieving overlap
(sub-3 A pairs 3.5 → 21.8 per ligand, `vina_score` -3.9 → +1.9).

Contact scale shows a genuine optimum (0.95 -5.56 / 1.00 -5.90 / **1.05 -6.12** /
1.10 -6.06 / 1.15 -5.05 / 1.25 -2.89 on 9 targets), and the relaxed poses stay in the
site: min ligand-pocket distance 2.52 → 3.20 A, contacts within 4.5 A 5.26 → 4.99 per
atom, pose movement 0.25 A RMSD.

Existing refiner checkpoints are a dead lever for `vina_score`; `refine_atom_bond_v1`
(the one already in use) is the best of the three. Note that omitting `--refiner` does
not disable the refiner — `ctbench.inference.generation._own_env` always sets
`SBDD_OWN_REFINE_CKPT` from the variant.

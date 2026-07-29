# Improving `vina_score` (pose as generated) — CrossDocked2020 100-pocket set

Session of 2026-07-28/29. Numbers are means over 100 targets x 100 molecules unless
a 25-target subset is stated; the summary table is `vina_score_summary.csv`.

## Result

| pipeline | vina_score | median | pb_valid | fair as-is? |
|---|---|---|---|---|
| baseline (session start) | -2.922 | -3.811 | 0.440 | — |
| **ML only** — distilled refiner x3 | **-5.025** | -5.242 | 0.254 | **yes** |
| physics — PATH2 relaxation, cs800 arm | -6.372 | -6.303 | 0.375 | no |
| **physics — PATH2 relaxation, x24 arm** | **-6.596** | -6.643 | 0.197 | no |

`vina_score` is the only Vina metric that responds to our coordinates: `vina_dock`
re-docks from a bare XYZ and discards both our poses and our SDF bond block
(writing real bond orders changed its mean by 0.001).

## What worked

**Restrain bond lengths and angles, free the torsions.** The pocket-overlap relief
(`scripts/relax_in_pocket.py`) needs an intramolecular restraint or it spreads the
atoms apart — PoseBusters validity 0.44 -> 0.11, strain tripled, and the apparent
score gain is an artifact. Restraining 1-2/1-3/1-4 distances fixes validity but
freezes the torsions, capping the score. Restraining only **1-2/1-3** keeps bond
lengths and angles exact while the dihedrals relax into the pocket, and score and
validity improve *together*: -6.613/pb 0.307 -> -6.889/pb 0.343 on the tuning
subset. The dilemma was never restraint strength; it was restraint topology.

Optimum: `--w-internal 100 --internal-path 2 --contact-scale 1.10-1.15
--w-tether 0.5 --w-uff 0 --pocket-source receptor`. Contact scale is an inverted U
(1.05 -6.50 / 1.10 -6.89 / 1.15 -6.93 / 1.20 -6.72 on 25 targets), and the relaxed
poses stay in the site: min ligand-pocket distance 2.52 -> 3.20 A while contacts
within 4.5 A hold at 5.26 -> 4.99 per atom, pose movement 0.25 A RMSD.

**Distillation is the refiner lever.** Training the e3nn refiner on
(generated pose -> relaxed pose) pairs — `scripts/build_distill_refine_set.py`,
then `scripts/apply_refiner_to_arm.py` at inference — beats the production
checkpoint by **+2.0 kcal/mol** on matched inputs (25 targets: -3.781 -> -5.814),
and needs no physics at inference. Three iterations of the network is the optimum;
five degrades.

## What did not work

| idea | result |
|---|---|
| attraction term (contact reward) | -6.46 -> -5.64 (w=1), +8.44 (w=5) |
| multi-start rigid search on the physics objective | -6.46 -> -6.16; our objective and Vina disagree about which basin is better |
| free tether (unrestrained placement) | -6.21 -> -6.01 |
| iterated relaxation (relax twice) | worse than once |
| vacuum UFF relaxation | -4.62 -> -1.43; relaxes toward gas-phase geometry and loses pocket fit |
| forcing >=28 heavy atoms at generation | arm mean *fell* to 17.96 atoms |
| lower temperature to make large molecules coherent | 20.4-20.6 atoms at T=0.70/0.85 — the limit is not temperature |
| lambda_bond 50/200 in distillation | diverges under iteration (pb 0.02-0.03); lambda_bond ~10 is the usable value |
| swapping to existing refiner checkpoints (geo_v1, place_v2) | -0.18 / -0.89 vs the current bond_v1 |

## Constraints found

**The LM's coherent-molecule ceiling is ~22 heavy atoms.** Forcing 42 generated
atoms still yields a largest perceived fragment of 22.3; the rest fragments off.
This is temperature-independent.

**Physics scales with molecule size, the network does not.** Relaxation improves
monotonically with size (22.4 -> 25.1 atoms: -6.37 -> -6.60) because hard
restraints hold the geometry regardless of atom count. The distilled network
degrades (-5.81 -> -4.99, pb 0.224 -> 0.105) because per-atom coordinate error
accumulates. The two pipelines therefore have *different* optimal arms — evaluating
the network on the physics-optimal arm understates it.

**Why the ML-only path stops 1.35 short.** A coordinate-regression refiner carries
~0.3 A per-atom error against a ~0.05 A tolerance for bond lengths, so it cannot
reproduce the hard geometric guarantee the restraint gives. Projecting the output
back onto the input's bond-length manifold restores validity but gives the score
back (-5.02 -> -4.49). The principled fix is to change the output parametrisation
from coordinates to **rigid-body transform + torsion angles**, which makes bond
lengths and angles structurally invariant — the same insight that made PATH2 work
on the physics side, moved into the network. Not implemented here.

**Distillation is capped by its teacher.** The first student (teacher: -5.5 to -5.8)
plateaued at -5.59, i.e. it had essentially caught its teacher. Rebuilding the set
on the PATH2 teacher (-6.5) is the right move but the student needs more epochs
than this session had: val RMSD 0.557 -> 0.554 between epochs 1 and 2, against
0.31 for the converged old-teacher student.

## joint vs separate

On the interaction-sensitive metrics the joint tokenizer wins: clash-free rate
0.697 vs 0.663 (p=0.020) and clash count 0.821 vs 0.894 (p=0.049) on matched arms,
consistent with the better interface lDDT seen in tokenizer reconstruction.
Separate wins on molecule size, QED and PoseBusters validity, and therefore on raw
`vina_score` after relaxation (-6.372 vs -6.299). A joint-centred claim should rest
on the interface metrics, not on raw Vina.

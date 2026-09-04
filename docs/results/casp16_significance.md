# CASP16 tokenize -> decode: ProLIT (joint, 3.0 ligand weighting, 250 epochs)

n = 303 CASP16 complexes, paired throughout. p-values Holm-corrected within
each block. **Mean favours** is what a summary table reports; **paired test
favours** is the direction the Hodges-Lehmann pseudomedian points, which is
what the signed-rank test locates. Where they disagree (⚠) the mean is
carried by a minority of samples and the p-value is evidence for the OTHER
side.

## Against the separate-tokenizer ablation (the paper's claim)

`separate_e250` is trained on the SAME recipe -- 250 epochs, distance map, constrained
balancing, held-out eval PDBs -- with 4096 codes per book so the pair matches the
joint arm's 8192 and its LM vocabulary. The ligand weight is joint-only because it
exists to share one book between two sources and there is nothing to share here.

| metric | rival | ProLIT | rival | mean diff | 95% CI | win rate | mean favours | paired test favours | Holm p |
|---|---|---|---|---|---|---|---|---|---|
| lDDT-PLI | separate | 0.945 | 0.926 | +0.019 | [+0.016, +0.022] | 80.7% | ProLIT | ProLIT | 7.6e-30 |
| Contact-F1 | separate | 0.688 | 0.666 | +0.022 | [+0.013, +0.032] | 59.1% | ProLIT | ProLIT | 1.5e-05 |
| Protein lDDT | separate | 0.964 | 0.887 | +0.077 | [+0.073, +0.081] | 99.3% | ProLIT | ProLIT | 6.1e-50 |
| Protein TM | separate | 0.881 | 0.753 | +0.128 | [+0.120, +0.137] | 99.3% | ProLIT | ProLIT | 7.6e-50 |
| Protein Kabsch (A) | separate | 0.417 | 0.754 | -0.338 | [-0.357, -0.319] | 99.7% | ProLIT | ProLIT | 2.7e-50 |
| PoseBusters validity | separate | 0.668 | 0.591 | +0.077 | [+0.027, +0.128] | 68.9% | ProLIT | ProLIT | 1.3e-02 |
| Ligand Kabsch (A) | separate | 0.329 | 0.352 | -0.024 | [-0.044, -0.003] | 51.7% | ProLIT | ProLIT | 6.5e-02 |
| SMILES match | separate | 0.416 | 0.446 | -0.030 | [-0.091, +0.030] | 44.6% | separate | separate | 3.8e-01 |

## Against ESM3 on the protein side

**Re-measured 2026-09-04 after fixing how the bench fed multi-chain targets to
ESM3** (`docs/results/2026-09-04_esm3_chain_handling.md`). The 47 sub-0.90-lDDT
complexes that used to carry this block were 57 two-chain targets handed to ESM3
as one butt-joined sequence, with author residue numbering that collided across
chains; they were our adapter, not ESM3. Numbers below are the corrected ones,
from `casp16_esm3_chainfix.parquet`.

| metric | rival | ProLIT | rival | mean diff | 95% CI | win rate | mean favours | paired test favours | Holm p |
|---|---|---|---|---|---|---|---|---|---|
| TM-score | ESM3 | 0.881 | 0.855 | +0.026 | [+0.008, +0.046] | 28.4% | ProLIT | ESM3 ⚠ | 1.4e-04 |
| lDDT | ESM3 | 0.964 | 0.965 | -0.001 | [-0.007, +0.006] | 27.7% | ESM3 | ESM3 | 2.1e-06 |
| Kabsch RMSD (A) | ESM3 | 0.417 | 0.611 | -0.194 | [-0.298, -0.101] | 29.7% | ProLIT | ESM3 ⚠ | 1.4e-04 |

Previously reported (superseded): ESM3 TM 0.811, lDDT 0.952, Kabsch 1.159.
**The lDDT mean win is gone**; TM and Kabsch keep a mean win but lose ~63% and
~75% of it and stay ⚠ — still carried by a minority.

That minority is now one identifiable failure mode. Splitting on whether the
pocket straddles a chain interface separates it perfectly:

| pocket | n | ProLIT Kabsch | ESM3 Kabsch | ProLIT wins | ProLIT lDDT | ESM3 lDDT | wins |
|---|---|---|---|---|---|---|---|
| within one chain | 260 | 0.457 | **0.344** | 18.1% | 0.955 | **0.983** | 15.8% |
| across two chains | 43 | **0.277** | 1.983 | **100.0%** | **0.996** | 0.876 | **100.0%** |

`StructureTokenDecoder.decode` forces `chain_id = zeros` ("not supported for
now", `esm/models/vqvae.py:375`), so ESM3 has no channel for relative chain
placement. **The protein claim is now: no chain-assembly failure mode, on ~24
near-duplicate dimer targets — not accuracy, and not general robustness.**

## The frame axis

| arm | frame | pose | lDDT-PLI | Contact-F1 | PB-valid | L-Kabsch | SMILES |
|---|---|---|---|---|---|---|---|
| `joint_e250_lig3` | shared pocket | free | 0.945 | 0.688 | 0.668 | 0.329 | 0.416 |
| `separate_e250` | shared pocket | free | 0.926 | 0.666 | 0.591 | 0.352 | 0.446 |
| `separate_e250_local_oracle` | own | free (oracle) | 0.946 | 0.707 | 0.772 | 0.234 | 0.523 |
| `separate_e250_local_3tok` | own | +39 bit | 0.942 | 0.698 | 0.772 | 0.234 | 0.523 |
| `separate_e250_local_2tok` | own | +26 bit | 0.866 | 0.563 | 0.772 | 0.234 | 0.523 |
| `separate_e250_local_1.5tok` | own | +20 bit | 0.725 | 0.391 | 0.772 | 0.234 | 0.523 |
| `separate_e250_local_1tok` | own | +13 bit | 0.485 | 0.178 | 0.772 | 0.234 | 0.523 |

A shared frame carries placement in every atom token and spends nothing on it. A
private frame spends all 13 bits per atom on shape instead, and the ligand comes
back markedly better -- PoseBusters 0.772 against 0.668, SMILES 0.523 against
0.416, Kabsch 0.234 against 0.329 -- but the placement then has to be sent
separately. **Three pose tokens buy the interface back; two do not** (lDDT-PLI
0.942 at 39 bits against 0.866 at 26). The ligand-internal columns do not move
with the pose budget, because the pose is where the ligand sits, not what shape
it is.

That gap is the largest single lever measured on ligand quality: +0.104
PoseBusters and +0.107 SMILES, against +0.084 SMILES for the entire ligand-weight
sweep from 3.0 to 8.3. The ligand's problem is that in the shared frame its bits
are split between where it is and what it is.


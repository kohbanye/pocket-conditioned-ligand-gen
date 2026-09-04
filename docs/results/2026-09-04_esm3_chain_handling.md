# ESM3's CASP16 outliers were our adapter, not ESM3

**The 47 "ESM3 collapses" complexes were 57 two-chain targets handed to ESM3 as
one butt-joined sequence.** Fixed in `recon_bench/adapters/esm3.py`
(`chain_break_layout`). Re-measured 2026-09-04, n=303, no failures.

Authoritative ESM3 dump: `benchmarks/recon-bench/results/casp16_esm3_chainfix.parquet`.
`casp16_esm3.parquet` is **superseded** — and its `own_vqvae` rows were never the
paper's arm anyway (Kabsch 0.851 against `joint_e250_lig3`'s 0.417); read ProLIT
from `casp16.arm-joint_e250_lig3.parquet`.

## The two defects

`datasets.casp16()` never sets `Sample.chain`, so `read_backbone` returns every
chain concatenated, and the adapter passed that straight through.

1. **No chain break.** The adapter padded BOS/EOS and nothing else. ESM3 encodes a
   complex as one sequence with a separator residue per boundary — NaN coords,
   `residue_index` -1, and a `CHAINBREAK` structure token (4100), which
   `esm/models/esm3.py:365-368` fills in wherever the sequence carries `|` (see
   `ProteinComplex.from_chains`). Without it the decoder folds chain B onto the
   end of chain A.
2. **Author residue numbers as `residue_index`.** Numbering restarts per chain, so
   610-residue L4001 sent 304 duplicate indices. `RelativePositionEmbedding` sees
   `key - query`, so residue 5 of chain A and residue 5 of chain B arrive at
   offset 0 — indistinguishable from a residue and itself. ESM3's own SDK path
   (`tokenize_structure` → `ProteinChain.from_atom37`) passes
   `arange(1, L+1)` and never has duplicates.

Both are ours. The PDB files are clean: chains contiguous, no insertion codes, no
duplicated altloc backbone atoms.

## What each fix is worth (57 two-chain samples, pocket scope)

| ESM3 input | Kabsch med | lDDT med | lDDT < 0.90 |
|---|---|---|---|
| butt-joined + author numbering (published) | 5.111 | 0.832 | 47/57 |
| + `arange` only | 1.337 | 0.878 | — |
| + chain break only | 3.105 | 0.860 | — |
| **+ both (adopted)** | **1.015** | **0.909** | 26/57 |
| each chain encoded alone (upper bound) | 0.336 | 0.978 | 0/57 |

**All five are identical on the 246 single-chain samples** (pocket Kabsch 0.336,
lDDT 0.982), so the protocol moves only the outlier population. Adopted variant =
ESM3's own multimer format + ESM3's own default numbering; it gives ESM3 one
global frame and one Kabsch, exactly what ProLIT gets. Per-chain encoding was
rejected as the headline: it zeroes inter-chain error by construction, a freedom
ProLIT does not get.

Per-series effect on ESM3 (pocket scope, median/mean Kabsch):

| series | n | before | after |
|---|---|---|---|
| L1 | 17 | 0.316 / 0.319 | 0.315 / 0.317 |
| L2 | 2 | 4.373 / 4.373 | 1.483 / 1.483 |
| L3 | 227 | 0.342 / 0.357 | 0.344 / 0.354 |
| L4 | 57 | 4.810 / 4.494 | **0.916 / 1.690** |
| all | 303 | 0.358 / **1.159** | 0.357 / **0.611** |

## What it does to the paper's claim

ProLIT `joint_e250_lig3` vs ESM3, paired, n=303, pocket scope, Holm-corrected:

| metric | ProLIT | ESM3 before | ESM3 after | mean diff after | 95% CI | win rate | paired favours |
|---|---|---|---|---|---|---|---|
| TM-score | 0.881 | 0.811 | 0.855 | +0.026 | [+0.008, +0.046] | 28.4% | ESM3 ⚠ |
| lDDT | 0.964 | 0.952 | **0.965** | **-0.001** | [-0.007, +0.006] | 27.7% | ESM3 |
| Kabsch (A) | 0.417 | 1.159 | 0.611 | -0.194 | [-0.298, -0.101] | 29.7% | ESM3 ⚠ |

- **The lDDT mean win is gone** — ESM3 now leads 0.965 to 0.964, and mean and
  paired test agree on ESM3. There is no protein-lDDT claim against ESM3 left.
- Kabsch and TM keep a mean win but lose ~75% and ~63% of it, and both stay ⚠:
  still carried by a minority.

## The claim that survives, and how narrow it is

The residual separates perfectly on **whether the pocket straddles the dimer
interface** — not on series, size, or ligand:

| pocket | n | ProLIT Kabsch | ESM3 Kabsch | ProLIT wins | ProLIT lDDT | ESM3 lDDT | wins |
|---|---|---|---|---|---|---|---|
| within one chain | 260 | 0.457 / 0.438 | **0.344 / 0.360** | 18.1% | 0.955 | **0.983** | 15.8% |
| across two chains | 43 | **0.277 / 0.285** | 1.983 / 2.128 | **100.0%** | **0.996** | 0.876 | **100.0%** |

Every ESM3 sample above 3 A is an interface pocket (15/43); none of the 260
single-chain pockets is. The mechanism is upstream and verified in source:
`StructureTokenDecoder.decode` forces `chain_id = zeros` with the comment "not
supported for now" (`esm/models/vqvae.py:375`), so the chain break is ESM3's only
channel for chain identity and it does not carry relative placement. We exhausted
what ESM3 exposes — per-chain `sequence_id` was also tried and made global
placement worse (Kabsch 15.976).

**Read this narrowly.** The 43 interface pockets come from ~24 L4 targets which
`project_casp16_set_is_four_proteins` shows are near-duplicates of very few
proteins, so the effective n is small and a 100% win rate is a statement about
one failure mode, not a robustness result. Claim it as "a pocket tokenizer that
reads the pocket as one atom set has no chain-assembly failure mode", and report
the 260-sample column next to it, where ESM3 is ahead.

## Not affected

- **FoldToken4** — reads the PDB itself through its own CLI; L4 pocket median
  0.66 A, no outlier population. No fix needed.
- **Bio2Token** — fed from ProLIT's pocket NPZ dumps, never sees a chain
  boundary. Structurally immune.
- **`separate` ablation and every interface metric** (lDDT-PLI, Contact-F1,
  PB-valid, SMILES) — different comparison, untouched.

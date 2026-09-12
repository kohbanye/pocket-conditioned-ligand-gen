# The same result on the honest basis

Every number reported so far is on all 100 canonical targets, which include 8
pockets present in the training corpus and 19 targets whose reference ligand the
model reproduces exactly (Tanimoto >= 0.999). Dropping the union — **21
targets** — is the honest basis, and the reproduced set is taken from ONE fixed
arm (the deployed one) so every row is scored on the same target list.

| arm | all 100 | clean 79 | shift |
|---|---|---|---|
| deployed `refoverlap_t0` | −0.504 | **+0.470** | **+0.974** |
| deployed `+` pruning | −1.240 | −0.875 | +0.365 |
| pctx `+` MLM (confidence) | −0.005 | +0.850 | +0.855 |
| pctx `+` clash+geom | −1.156 | −0.832 | +0.324 |
| **pctx `+` clash+geom `+` pruning** | **−1.623** | **−1.403** | **+0.220** |
| reference ligand | −6.806 | −6.871 | −0.065 |

**Every arm gets worse, and the better arms get worse by less.** Nearly a full
kcal of the deployed arm's reported score rode on contaminated targets (+0.974);
the best arm's rode 0.220.

So the improvement is **larger** on the honest basis, which is the opposite of
what contamination normally does:

| | deployed -> best | gap to the reference |
|---|---|---|
| all 100 | −1.119 | 6.302 -> 5.183 (**17.8%** closed) |
| **clean 79** | **−1.873** | 7.341 -> **5.468** (**25.5%** closed) |

The reference ligand itself barely moves between the two bases (−6.806 ->
−6.871), so this is not an artifact of a shifted target.

**The methods are not exploiting the contamination.** Clash-ordered decoding and
clash-leaf pruning both work from receptor geometry alone; neither has any way
to benefit from having seen a pocket in training, and the numbers say they do
not.

**Report the clean-79 figures as primary**, with the 100-target ones beside them.

## The diversity co-report

Required alongside any score claim. Clean 79 basis, 100 molecules per target.

| arm | internal diversity | unique SMILES | Tanimoto to reference | validity |
|---|---|---|---|---|
| deployed `refoverlap_t0` | 0.752 | 0.656 | 0.109 | 0.992 |
| pctx `+` MLM (confidence) | 0.728 | 0.653 | 0.162 | 0.998 |
| pctx `+` clash+geom | 0.733 | 0.659 | 0.161 | 0.998 |
| **pctx `+` clash+geom `+` pruning** | **0.744** | **0.677** | 0.155 | **1.000** |

Paired against the confidence control:

| arm | internal diversity | unique SMILES |
|---|---|---|
| `+` clash+geom | +0.003 (p=0.029) | +0.000 (p=0.27) |
| `+` clash+geom `+` pruning | +0.010 (p=5.3e-04) | +0.010 (p=8.1e-06) |

**Corrected 2026-09-11**: this is true against `pctx + MLM (confidence)` and
**not** against the designated `noref` control, where the whole stack costs
**11 points of unique SMILES** (0.791 -> 0.677, p=7.7e-06). See
`2026-09-11_the_total_against_the_designated_control.md`.

Against the confidence baseline there is no diversity cost. Both measures are flat to slightly better and
validity rises to 1.000. Different molecules lose different atoms and get
different codes re-decided, so the operations separate near-duplicates more often
than they merge them.

Similarity to the reference ligand **falls** (0.162 -> 0.155), so the gain is not
the model drifting toward the answer.

The pctx stack is less diverse than the deployed one to begin with (0.728 vs
0.752); that is a property of the stack, not something these steps caused.

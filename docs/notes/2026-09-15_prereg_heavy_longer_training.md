# Pre-registration: the heavy recipe is under-trained at 20 epochs

2026-09-15 ~18:55 JST. Written BEFORE any head of this arm exists. Same
fixed-size design as the heavy and all-atom rounds.

## The evidence that prompted it

Best-validation epoch and stopping epoch, `--max-epochs 20 --early-stop-patience 4`:

| arm | best epoch | stopped at |
|---|---|---|
| `aff_e250_unionheavy_s26` | 14 | 18 |
| `aff_e250_unionheavy_s27` | 13 | 17 |
| `aff_e250_unionheavy_s28` | **18** | 19 |
| `aff_e250_unionnccorr_s18` | 9 | 13 |
| `aff_e250_unionnccorr_s19` | 5 | 9 |
| `aff_e250_unionr8_s18` | 5 | 9 |
| `aff_e250_unionallatoms_s34` | 3 | 7 |

The ca recipes peak at epochs 3-9 and stop well inside the cap. The heavy recipe
peaks at 13-18 and stops at 17-19, i.e. **against the cap**, with one seed still
improving at epoch 18. Its documents are ~1.5x longer (mean 367 tokens against
247), so each epoch shows the head a harder problem; 20 epochs appears to be the
binding constraint for this recipe and not for the others.

Stated now so it cannot be reinterpreted later: **validation losses are not
comparable across these arms** (the corr recipe mixes a correlation term into
its loss, and the heavy recipe has its own 878-document validation split). Only
the epoch counts above are used as evidence.

## Arms

* **Treatment** `e250_unionheavy_long` (NEW, 8 seeds 50-57): corpus
  `data/lm_tokens_e250aff_union_heavy`, `--max-epochs 40`, everything else
  identical to `e250_unionheavy` (`--label-cap 13 --early-stop-patience 4
  --block-size 640`, ligand readout, heavy pocket at inference).
* **Baseline mix (16):** `e250_unionnc_corr_v2` + `e250_unionr8_ens_v2`, seeds 18-25.
* **Control (24):** baseline mix + the 8 standard heavy seeds (26-33).

## Predictions (f32 basis)

* **L1.** R(16 fresh + 8 heavy-long) - R(16 fresh + 8 heavy-standard) >= **+0.004**.
* **L2.** rho(16 fresh + 8 heavy-long) >= rho(16 fresh + 8 heavy-standard) - 0.005.

**Works only if L1 and L2 hold.** f16 reported beside.

**Void condition, fixed now:** if the treatment heads stop *before* epoch 20
(i.e. early stopping fires inside the old cap), the premise is false, the round
is reported as void rather than as a refutation, and the epochs reached are
recorded.

## Cost

8 heads (gpu_1, 2:00 -- 20 epochs took ~20 min, so 40 needs the larger slot)
~= 3.2 pt + 16 inferences ~= 1.6 pt ~= **4.8 pt**, ledger to about 139 of 200.

# Overnight summary, 2026-09-15 23:00 → 2026-09-16 04:30 JST

Five levers tested, all refuted. **Zero compute spent after the ceiling
calculation.** Ledger unchanged at **179.18/200 pt** (20.82 remaining). Queue
empty, no jobs running, no failures.

## Goal status

| condition | target | result | verdict |
|---|---|---|---|
| W1 | beat GenScore x1 on **both** R and rho | R 0.8229 vs 0.8127, rho 0.7316 vs 0.7281 | **TRUE** |
| W2 | beat GenScore x3 on both | R 0.8229 vs 0.8387 | FALSE |
| W3 | R > 0.83 | 0.8229 | FALSE |

Re-verified this session by running `mix36_verdict.py` (the pre-registration
applied mechanically), not quoted from an earlier session:

| | R | rho | R clean (n=266) | rho clean |
|---|---:|---:|---:|---:|
| **ours: 3 recipes, 36 models** | **0.8229** | **0.7316** | **0.8261** | **0.7643** |
| GenScore, 1 released checkpoint | 0.8127 | 0.7281 | 0.8184 | 0.7589 |
| GenScore, 3 released checkpoints | 0.8387 | 0.7211 | 0.8421 | 0.7411 |
| our mean single seed | 0.7920 | 0.6718 | | |

W1 survives the twin screen on both metrics. The GenScore ensemble reproduced to
max |diff| 3.16e-05.

## Two caveats the write-up must carry

1. **Cost asymmetry.** 36 of our models against 1 of theirs is not equal cost;
   the pre-registration prints this itself. W1 holds *as registered*, but the
   honest framing is "36-model mix vs single released checkpoint".
2. **Tokenizer leak, unresolved.** `vq_e250_lig3` underlies every e250 arm and
   its CASF membership cannot be verified from the checkpoint (the exclusion
   flag is opt-in and not stored; the same-lineage V1 had 169/285 CASF complexes
   on its train side). It is *unsupervised* exposure — no affinity labels — and
   CASF labels were held out of head training. New evidence tonight: **MLM
   familiarity does not predict accuracy** (corr(pll, |residual|) = -0.056;
   most-familiar quartile |resid| 1.045 vs least-familiar 1.094), so
   memorisation did not transfer. Swapping to the leak-free `vq_ctrl_p3` would
   change the vocabulary and require retraining the MLM and all 36 heads — far
   beyond 20.82 pt.

## Configuration choice for the paper

The **24-model mix has higher R (0.8244) but lower rho (0.7228)**, which fails
W1 on rho. The 36-model mix is the only configuration satisfying both. Choosing
between them after seeing results would be selection on the outcome; the 36-mix
is the pre-registered answer.

## What was refuted tonight

| lever | result |
|---|---|
| ensemble disagreement as an error proxy | corr(disagree, \|bias\|) +0.234; complex-level +0.160 |
| **target-offset predictor** | nested LOO rho **+0.088**, corrected R 0.8215, **p = 0.527** |
| within-target compression rescale | our slope 1.090 vs GenScore **1.115** — they are more compressed |
| implementation defect | ours 12 outliers (\|resid\|>2.5) vs GenScore 14 — we have fewer |
| label noise | 72/15,351 (**0.47%**) outside pK [2,13] |
| data scale (rediscovered) | +73% corpus → dR **-0.0055**, drho **-0.0228** (refuted 2026-09-14) |

## The structural picture, now enumerated rather than guessed

* **Target bias is shared, not ours-specific-and-fixable.** 83% of per-target
  bias variance is common to every recipe family. Our bias sd is **0.629** vs
  GenScore's **0.715** — *we are less target-biased than GenScore*. An earlier
  framing of our bias as pathological was wrong.
* **The bias is concentrated but unpredictable.** 13 targets are coherent
  offsets carrying 56% of squared bias (removing them → R 0.8512; top 5 →
  0.8442; all → 0.8731). Required predictor correlation is only rho ~ 0.27. The
  honest achievable correlation is **0.088**. 0 of 200 null draws reached 0.83.
* **No covariate explains it.** 18 label-free covariates, nested LOO: noise.
  `pred_mean` alone gives -0.078, so it is not a recalibration artefact either.
* **Data scale is not the constraint.** +10,523 complexes made it *worse*, and
  the weak (between-target) half moved backwards.
* **Representation change does not decorrelate it.** The v1 generation — a
  different VQ *and* a different MLM — correlates +0.8519 with the e250 shared
  bias, and mixing it in moves R by +0.0003.

**Conclusion: R > 0.83 is not reachable by any lever available in this
architecture family, not merely unaffordable within the remaining budget.**

## Where the deficit actually is: per-model quality, not ensembling

| | single | ensembled | gain from ensembling |
|---|---:|---:|---:|
| GenScore | **0.8127** | 0.8387 (3 checkpoints) | +0.026 |
| ours | **0.7920** | 0.8229 (36 models) | +0.031 |

Our ensembling is *better* than theirs (+0.031 vs +0.026). We start **0.0207
lower per model**, and 36 models cannot close what 3 of theirs clear. Combining
their per-model quality with our ensembling gain would land near 0.844 — above
the bar. So the whole R deficit is per-model quality.

Stated symmetrically, because it cuts our way too: **on rho we are the best of
the three** — 0.7316 against GenScore's 0.7281 (x1) and 0.7211 (x3); their
ensembling *costs* them rho. The loss is confined to R against their
3-checkpoint ensemble.

**What causes the per-model gap is not established.** The competitive methods
all use explicit interaction featurisation, which this project deliberately
excludes, and our target bias correlates only +0.46 with theirs — but that is
circumstantial. No experiment here isolates featurisation as the cause, so it
should be written as a hypothesis, not a finding.

## Honest note on this session's judgment

Four times tonight I announced a promising lead and the honest test killed it:
the ceiling calculation read as "reachable"; the offset predictor at "R 0.8323"
(lambda selected by reading the LOO column — selection on the test statistic);
within-target compression (circular — I picked the worst-scatter targets, then
called them compressed); and the IC50 corpus (already refuted two days earlier,
findable by one grep). Every measurement that survived scrutiny was negative.
The fourth was fully avoidable and is now recorded as a standing rule: grep
`docs/results/` and `variants.py` **before** reasoning about a lever.

## Recommendation

Report W1 with both caveats stated. Do not spend the remaining 20.82 pt on the
untested covariates (MLM pooled embeddings at 768 dims on 57 targets; VQ
reconstruction error, confounded by the same CASF membership question) — both
are predicted to fail by tonight's measurements, which showed 18 covariates
yielding rho 0.088.

---

## Repo health check (2026-09-16 ~04:45 JST)

Ran the three CI commands against the working tree, which carries ~35
uncommitted files from this campaign (LoRA, `layer_mix`, distillation, new
`config.py` fields).

| check | result |
|---|---|
| `ruff check` (283 tracked .py, `third_party` excluded) | **clean** — "All checks passed!" |
| `ty check src` | **exit 1**, 7 errors + 2 warnings |
| `pytest` | blocked locally — see below |

### ty: 6 of 7 errors are newly introduced

Established by comparing against committed blobs (`git show HEAD:<path>`) — no
stash, checkout or worktree, so the working tree was never touched.

| diagnostic | at HEAD | new? |
|---|---|---|
| `lora.py:30,48,49` | file absent | **new** |
| `rescore_module.py:112,115,116` (`readout`/`layer_mix`/`lora_rank`) | 0 occurrences of each | **new** |
| `mlm_module.py:57` (`self.trainer.datamodule`) | 0 occurrences | **new** |
| `rescore_module.py:186` (`requires_grad_(requires_grad=True)`) | present, `_freeze_encoder` unchanged | likely pre-existing |
| `vqvae.py:847` (`balanced_chem_loss`, warning) | file unmodified | pre-existing |

**CI's `Types` step (`uv run --frozen ty check src`, `if: !cancelled()`, no
continue-on-error) would fail if this tree is committed as-is.** `main` is
probably green; proving that needs `ty` against a pristine HEAD checkout (a
worktree plus a dependency install), which was not done.

The fixes appear safe and none touch `state_dict` keys: `requires_grad_(False)`
is equivalent to the flagged keyword form, and the `rescore_module.py`
diagnostics are ty misreading LightningModule's custom `__setattr__` on plain
config attributes. Not applied — unrequested source edits to files underpinning
36 trained heads should be the author's call.

### pytest is blocked by an untracked test, not by the library

`tests/test_genscore_matched_train.py` imports `genscore_matched_train` from
`experiments/affinity_loop_20260914/`, which imports `torch_geometric` (absent
from the main venv). One uncollectable module aborts the whole run.

**Both files are untracked**, so CI never sees them and this is local-only. No
tracked test imports from `experiments/`. A re-run ignoring that module is in
flight.

### Method note: two false passes

The first two pytest invocations reported "exit code 0" while running **zero**
tests — once dying on an unsupported `--timeout` flag, once interrupted at
collection. The shell is zsh, where `${PIPESTATUS[0]}` is empty (`$pipestatus`),
and a trailing `| tail` masked the real status. Also `ruff check .` from the
repo root reported **55,915** errors by scanning vendored venvs and
`experiments/*/snapshot_*/` copies of `src/` that CI never sees. Check the
artifact, never the exit code; scope the lint to tracked files.

### The hanging test: `report.scoreboard` is O(N^2) in the number of arms

`benchmarks/affinity-bench/tests/test_reproduce.py::test_scoreboard_states_the_goal`
(test #464 of 508) ran for **66 minutes** at 171% CPU before being killed.
Diagnosis, verified in the source rather than inferred:

```python
def beats_reference(preds, method, reference="GenScore", ...):
    metrics = affinity_metrics(preds, pred_col)         # over ALL methods
    sig     = affinity_pairwise(preds, reference, ...)  # over ALL methods
    return pd.Series({... only `method`'s row ...})     # discards the rest
```

`scoreboard` calls it once per arm:
`{m: beats_reference(preds, m, REFERENCE) for m in ours}`.

`methods()` loads **one protocol at a time** (`_f16`), so the 497 directories in
`results/` resolve to **250 methods / 245 `OURS` arms** (the rest are the `_f32`
variants). Each of those 245 arms recomputes the full metrics table and the full
pairwise battery over all 245 — **~60,000 Williams/Steiger + Wilcoxon tests**
per `scoreboard()` call, all but one row of each discarded.

**Measured, not inferred.** Per-arm cost is linear in N (0.452 s at N=5, 0.793 at
10, 1.469 at 20, 2.981 at 40, 5.854 at 80 — slope **72.3 ms/arm**), which makes
the per-arm loop quadratic overall. Extrapolating to N=245: 17.7 s per arm x 245
arms = **72 minutes predicted**, against **66 minutes observed** before the run
was killed. Same order; the mechanism is confirmed.

**It is a scaling regression, not a broken test.** `results/` dates from Sep 13
with a handful of arms, where N^2 was free; this campaign grew it to 497.

Confirming evidence: `report.comparison(_RESULTS)` calls `methods` + `metrics` +
`pairwise` exactly **once** and completed as test #463, immediately before
`scoreboard` hung as #464.

**Not a GPU problem.** The process held `/dev/nvidia0`, but those fds are left
over from earlier torch tests in the same process; the work is pure pandas/numpy
(171% CPU, state `R`).

**Fix (behaviour-preserving):** hoist `affinity_metrics` and `affinity_pairwise`
out of the per-arm comprehension, compute once, index per arm. O(N^2) -> O(N),
identical output. Not applied — unrequested source edit.

**Correction to an earlier claim in this file.** I wrote that a long-running test
"would also stall CI". That is **wrong**: the module carries
`pytestmark = pytest.mark.skipif(not _OURS.exists(), ...)` and the dumps are
git-ignored, so it skips in CI. Like the `torch_geometric` collection error, this
is local-only. It does, however, make the local suite unusable past ~90%.

Test-order identification was valid: only `pytest` itself is installed (no
`pytest-randomly`/`-xdist`/`-random-order`), so collection order == execution
order, and 508 collected matches 432 dots at the 85% mark.

### The repo's own goal tripwires cannot register W1

Both tests that exist to detect "have we beaten GenScore?" **passed** in tonight's
run, and neither can fire on the result we actually have.

* `test_we_do_not_yet_beat_genscore` asserts `d_scoring_R < 0` **and**
  `d_ranking_rho < 0`. Its `_ours()` reads `results/v1_kdki_mean/kdki-mean.csv`
  — the V1 single head at R 0.7711 / rho 0.6474 — not the 36-model mix. Its own
  docstring says "When this fails because both deltas turned positive, that IS
  the result."
* `test_scoreboard_states_the_goal` asserts `not board["beats_both"].any()`, and
  `scoreboard` is built from `methods()` -> `discover_arms()`, which enumerates
  **directories** (`d.is_dir()`).

The 36-model mix is a z-sum computed inside the verdict scripts. It has no
directory in `results/`, so the reporting layer cannot see it. Confirmed: no
mix/zsum artifact exists there (the only "mix" matches are `layermix`, a recipe
name).

**Verified per-arm picture (f16, the protocol `methods()` loads):**

| | R | rho |
|---|---:|---:|
| best single arm, `e250_unionnc_kd_s126_f16` | 0.8183 | 0.6895 |
| `e250_unionnc_layermix_lr_s92_f16` | 0.8137 | 0.6649 |
| GenScore | 0.8127 | 0.7281 |
| **the 36-model mix (not visible to these tests)** | **0.8229** | **0.7316** |

**0 of 245 `OURS` arms beat GenScore on both metrics**, so the tripwires report
the truth about individual arms while being blind to the mix.

Note the best single arm comes from the **KD family, a refuted lever**: it clears
GenScore on R but collapses on rho. A single arm's R means little when its rho
falls 0.039 short — precisely the cherry-pick the pre-registration exists to
prevent.

**Actionable:** if W1 is to be recorded in the codebase rather than only in
`docs/`, either dump the mix as an arm directory so `discover_arms` finds it, or
re-point the assertions at the mix. Not applied — unrequested source edit.

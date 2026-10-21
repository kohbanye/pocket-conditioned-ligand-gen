# Where we lose, from scratch, and a pre-registered fix (2026-09-14 ~20:40)

Written after the pre-registered headline failed and ten levers had been
refuted, following the user's instruction to set prior narratives aside and
reason from first principles. Everything under "Diagnosis" was measured on CASF
and is therefore exploratory. Everything under "Pre-registration" was fixed
BEFORE the arms below were implemented or trained.

## Diagnosis

### 1. R decomposes exactly into a between-target and a within-target part

CASF-2016 is 57 targets x 5 ligands. Label variance is **37% between targets,
63% within**. Pearson R over 285 = (cov_between + cov_within) / (sd_p sd_y),
where "between" is each target's mean and "within" the deviation from it.

| method (f32 where ours) | R | from between | from within | R of 57 target means |
|---|---:|---:|---:|---:|
| ours, headline ensemble | 0.8100 | **0.3704** | 0.4396 | 0.854 |
| GenScore | 0.8127 | 0.3682 | 0.4445 | 0.848 |
| GIGN mean-of-3 | 0.8550 | **0.4128** | 0.4422 | 0.911 |
| GIGN single (3 checkpoints) | 0.832-0.845 | 0.399-0.413 | 0.432-0.438 | 0.895-0.918 |
| OnionNet-2 | 0.8513 | **0.4290** | 0.4222 | 0.938 |

**Against GIGN and OnionNet-2 the R deficit is almost entirely between targets**
(-0.042, -0.059). Within targets we are level with GIGN (-0.003) and ahead of
OnionNet-2. The loss is predicting which *protein* binds strongly on average,
not ranking ligands on a fixed pocket. Swapping in GIGN's between-target part
while keeping ours for within-target gives R 0.8336. Our error is not
outlier-driven (worst-10% share of squared residual 0.43 vs GIGN 0.55).

### 2. It is not target coverage

GIGN's training set covers 41 of 57 CASF targets by exact chain sequence, ours
39 (3 differ). The winners predict target-level affinity **better on targets
they never trained on**: between-target |residual| GIGN 0.288 uncovered vs 0.356
covered; OnionNet-2 0.206 vs 0.308. Ours is flat (0.373 vs 0.395). So their
advantage is genuine generalization of pocket-level affinity, not memorized
target means, and more PDBbind coverage (+3 targets) is not the fix.

### 3. The signal is already in our representation; the readout discards it

Same corpus (`union_noconf`), same recipe, only the readout differs (f32 family
ensembles; seed means agree):

| readout | R | from between | from within |
|---|---:|---:|---:|
| `ligand` (mean over ligand atoms) | 0.8101 | 0.3801 | **0.4301** |
| `allatoms` (mean over protein + ligand atoms) | 0.8003 | **0.4047** | 0.3957 |
| `balanced` (average of protein-mean and ligand-mean vectors) | 0.8067 | 0.3755 | 0.4313 |

Pooling protein atoms raises the between-target part to GIGN's single-model
level, but because pocket atoms outnumber ligand atoms it dilutes the
ligand-specific signal that orders ligands within a target. Pooling ligand
atoms only does the reverse. `balanced` averages the two vectors into one, so
the pocket signal is washed out again. **No existing readout keeps both.**

## Logical consequence

Keep the two pooled vectors separate: **concatenate [protein-mean, ligand-mean]**
into the head. The head can then use pocket features for target-level affinity
and ligand features for within-target discrimination without either diluting
the other. If both components are realized, R ~ 0.4047 + 0.4301 = **0.835**,
consistent with the independent swap-test estimate (0.832-0.834).

This keeps the tokenizer and LM untouched and the head a single MLP; the only
change is the first layer's input width (one extra hidden-size block). That is
stated plainly because the head is meant to stay simple.

## Pre-registration

Fixed before implementation and before any head is trained.

**Arms** (seeds 7, 11, 12, 13; `pipelines/train/scoring_head.py` with exactly
the recipe of the control except `--readout concat`; backbone
`mlm_e250lig3` e02; f16 and f32 inference):

| family | corpus | control (same driver, same seeds) |
|---|---|---|
| `e250_unionnc_concat` | `data/lm_tokens_e250aff_union_noconf` | `e250_union_noconf` (readout `ligand`) |
| `e250_kdkictl_concat` | `data/lm_tokens_e250aff_kdki_ctl` | `e250_aff_kdki_ctl` (readout `ligand`) |

Two corpora because a readout effect judged on one family is exactly the error
that sank f32 today. Same driver as each control, so equal seed numbers do pair
trajectories (unlike the dev-runner comparison).

**Predictions** (f32 family ensemble, decomposition as above):

* P1. between-target part >= 0.395 on BOTH corpora (i.e. near `allatoms`).
* P2. within-target part >= 0.420 on BOTH corpora (i.e. near `ligand`).
* P3. total R above its control on >= 3/4 paired seeds on BOTH corpora.

**Lever verdict:** the concat readout counts as a working lever only if P1, P2
and P3 all hold on both corpora. Anything less is reported as a refutation or
a partial result, with its numbers.

**What this does not do:** it does not reopen the headline verdict. The
pre-registered headline (`docs/results/2026-09-14_prereg_headline_verdict.md`)
failed and stands. If the lever works, a new headline requires a new
pre-registration written before its ensemble is scored.

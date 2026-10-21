# prolit-pose-rescoring-bench

Evaluation & ablation benchmark for the protein–ligand **complex tokenizer**
trained in the sibling repo `../pocket-conditioned-ligand-gen` (the "source
repo"). The source repo holds only the models (tokenizer VQ-VAE + downstream
LM/MLM/heads and their checkpoints); **all inference, evaluation, statistics,
and figures live here.**

Two questions, on all three downstream tasks (ligand generation, pose
rescoring, affinity prediction):

1. **Comparison vs existing methods** — reproduced here (DiffGui/TargetDiff/
   DiffSBDD; RTMScore/GenScore/Vina; Boltz-2), judged by significance, not just
   point estimates.
2. **Ablation** — does a **jointly-trained** complex tokenizer beat separately
   trained **protein-only** and **ligand-only** tokenizers? Same three tasks,
   same protocol, paired tests + Holm correction against `joint`.

## Layout

```
pose_rescoring_bench/
  config.py, variants.py     # dataclass config; variant -> per-task checkpoints
  io_dumps.py                # canonical per-sample dump schemas
  metrics/{rescoring,affinity,generation}.py   # metrics ported from the paper notebooks
  stats.py                   # ttest_rel / Steiger / Wilcoxon / McNemar / bootstrap / Holm
  aggregate.py, report.py    # dumps -> comparison & ablation tables + significance
  inference/                 # load source-repo models, run tasks -> dumps (GPU)
  adapters/                  # run each published baseline here, in its own env
  baselines/                 # legacy: collect results a sibling checkout produced
  paths.py                   # baseline sources, interpreters, weights
  plotting.py                # figures
scripts/                     # argparse entrypoints + jobs/ (SGE qsub templates)
results/                     # per-sample dumps: seeded + generated (see below)
tests/                       # metric/stats unit tests + numeric reproduction tests
```

`results/<task>/<method_or_variant>/…` — e.g. `results/affinity/joint/*.csv`
(our ensemble heads), `results/affinity/genscore/scoring.csv` (baseline),
`results/rescoring/{joint,rtmscore,genscore,vina}/…`,
`results/generation/{joint,baselines}/…`.

## Analysis (no GPU)

```sh
uv sync
uv run python scripts/collect_baselines.py   # seed results/ from sibling repos
uv run python scripts/make_tables.py         # comparison + ablation tables -> results/tables/
uv run python scripts/make_figures.py        # figures -> results/figures/
uv run pytest                                # unit + numeric-reproduction tests
```

The reproduction tests confirm the analysis layer reproduces the source repo's
published numbers (e.g. affinity GenScore R=0.816 / ours=0.790; the full pose
docking-power table; generation vs DiffGui paired t p=0.39).

The arm the pose table reports is `variants.PAPER_ARM`, in one place, because it
had drifted: `report.py` hard-coded `joint` (the first-generation `xzkjxu9q`
tokenizer, DP@2A 89.5) while the reproduction tests pinned `e250_div` (95.4), so
"the paper's table" meant two different things depending on which file you read.

`_rescoring_variant_heads` z-sums **every** `*.csv` in a variant's directory, so
a dump written under a different protocol (a different frame count, say) must go
to its own `results*` tree -- dropping it beside the others silently turns a
single head into an ensemble.

## Baselines: measured here, not copied

Each published baseline is an adapter under `pose_rescoring_bench/adapters/`
that scores the CASF-2016 decoy stacks with the method's own released weights,
in the method's own environment, on the same poses and RMSD reference our arms
use. No number in the table is copied from a paper.

```sh
uv run python scripts/measure_baselines.py --baseline rtmscore
uv run python scripts/measure_baselines.py --baseline genscore
uv run python scripts/measure_baselines.py --baseline gnina
uv run python scripts/measure_baselines.py --baseline vina --workers 150
uv run python scripts/measure_baselines.py --baseline rtmscore --pdbids 1a30  # smoke
```

A measured dump lands at `results/rescoring/<name>/pose_scores_measured.csv`,
beside rather than over the collected `pose_scores.csv`: replacing a published
row is a decision about the paper, not a side effect of re-running a script.

| baseline | environment | weights |
|---|---|---|
| `rtmscore` | DGL; `paths.RTMSCORE_PYTHON` | `third_party/RTMScore/trained_models` |
| `genscore` | PyTorch-Geometric; `paths.GENSCORE_PYTHON` | `third_party/GenScore/trained_models` |
| `gnina` | Apptainer container (`scripts/fetch_gnina.sh`) | inside the binary |
| `vina` | this bench's env + the `vina`/`obabel` executables | none (empirical) |

RTMScore and GenScore are Python 3.8 conda environments, several releases behind
this bench, so the driver they share (`scripts/casf_pose_score_cli.py`) avoids
anything newer. gnina's released "static" binary needs a newer glibc than this
cluster has, hence the container.

**Inside a job, point the tools at themselves.** Job scripts deliberately do not
source the interactive rc file, so a tool installed under `$HOME` is not on
`PATH` and `prolit.external_tools` cannot find it:

```sh
python jobs/submit.py --name rtm_pose --resource gpu_1 --hours 3 \
    --env PROLIT_OBABEL=/path/to/obabel -- \
    benchmarks/pose-rescoring-bench/scripts/measure_baselines.py --baseline rtmscore
```

## Inference (GPU, qsub)

`uv sync` installs the source repo as an editable dependency, so `import prolit.*`
resolves. Then run per variant (SGE templates under `scripts/jobs/`):

```sh
VARIANT=joint qsub scripts/jobs/infer_rescoring.sh
VARIANT=joint qsub scripts/jobs/infer_affinity.sh
VARIANT=joint qsub scripts/jobs/infer_generation.sh
```

Dumps land in `results/<task>/<variant>/`; re-run `make_tables.py` /
`make_figures.py` to refresh.

## Ablation status

Only the `joint` variant is trained today. `protein_only` and `ligand_only` are
registered in `pose_rescoring_bench/variants.py` with empty checkpoint slots; fill them once
the source repo trains those tokenizers (+ their downstream models), then run the
same inference jobs. The ablation tables/figures populate automatically.

## Tooling

uv + Ruff (`select = ["ALL"]`) + `ty`, Python 3.12, PyTorch Lightning, stdlib
`@dataclass` config — matching the boilerplate/source-repo conventions.

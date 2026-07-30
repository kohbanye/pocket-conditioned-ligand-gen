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
  baselines/                 # collect + rerun existing-method results (code lives here)
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

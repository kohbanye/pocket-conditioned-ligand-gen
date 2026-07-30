# benchmarks/

Three benchmarks, one per result table in the paper. Each answers a different
question about the same tokenizer, so each has its own data, its own baselines
and its own execution environment — but they must agree on what a tokenizer arm
*is*, which is what `common/` exists for.

| directory | paper table | question | baselines |
|---|---|---|---|
| `recon-bench/` | Table 1 | how faithfully does a tokenizer reconstruct a complex? | ESM3, FoldToken4, Token-Mol, Bio2Token, ConfSeq |
| `pose-rescoring-bench/` | Table 2 | how well do ProLIT tokens rank binding poses? (+ affinity, + the ablation across all three tasks) | RTMScore, GenScore, AutoDock Vina, DeepRMSD, Boltz-2 |
| `sbdd-bench/` | Table 3 | how good are ligands generated for a pocket? | DiffSBDD, TargetDiff, DiffGui |
| `common/` | — | the shared arm registry and significance tests | — |

## Environments

`common`, `pose-rescoring-bench` and `sbdd-bench` are members of the root uv workspace and
share the repository's `.venv`.

**`recon-bench` is not**, and this is deliberate: it runs ESM3 in-process, and ESM3
pins a fork of `transformers` that would replace the `transformers` 5.x the
ProLIT language models are built on. A uv workspace resolves one version per
package, so membership would silently downgrade the model library. `recon-bench`
keeps its own project and lockfile:

```sh
cd benchmarks/recon-bench && uv sync            # its own .venv
```

Its FoldToken and Bio2Token adapters go further still and run in dedicated
venvs (`.venv-foldtoken`, `.venv-bio2token`) as subprocesses, because those
pin exact CUDA-coupled torch builds.

The generative baselines under `sbdd-bench` are the same story in conda form:
each generates in its own environment and writes a plain `generated.sdf`, which
the benchmark environment then scores. See `sbdd_bench/README.md`.

## The arm registry

`common/src/prolit_bench/variants.py` defines which runs, normalization
statistics and codebook sizes each arm (`joint`, `separate`, `separate_4096`)
means. Read its module docstring before changing anything there — it documents a
real discrepancy between the tables that is recorded rather than hidden, and
`test_variant_agreement.py` fails if the benchmarks drift apart again.

## Running the analysis (no GPU)

```sh
uv run pytest benchmarks                    # metric, stats and reproduction tests
uv run python benchmarks/pose-rescoring-bench/scripts/make_tables.py
uv run python benchmarks/pose-rescoring-bench/scripts/make_figures.py
```

`results/` is git-ignored: only code is tracked, and anything a run reproduces
stays local. The reproduction tests assert that the analysis layer still
produces the published numbers from whatever dumps are present, and skip when
they are not — so a fresh clone runs green with them skipped. Which checkpoint
produced which published number is recorded in `docs/results/`, which is
untracked and lives beside the checkpoints.

One test is an expected failure rather than a skip: no TargetDiff dump anywhere
reproduces the generation table's -4.76 (see
`pose_rescoring_bench/tests/test_reproduce_generation.py`).

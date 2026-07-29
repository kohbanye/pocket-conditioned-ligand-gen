# benchmarks/

Three benchmarks, one per result table in the paper. Each answers a different
question about the same tokenizer, so each has its own data, its own baselines
and its own execution environment — but they must agree on what a tokenizer arm
*is*, which is what `common/` exists for.

| directory | paper table | question | baselines |
|---|---|---|---|
| `plbench/` | Table 1 | how faithfully does a tokenizer reconstruct a complex? | ESM3, FoldToken4, Token-Mol, Bio2Token, ConfSeq |
| `ctbench/` | Table 2 | how well do ProLIT tokens rank binding poses? (+ affinity, + the ablation across all three tasks) | RTMScore, GenScore, AutoDock Vina, DeepRMSD, Boltz-2 |
| `sbddbench/` | Table 3 | how good are ligands generated for a pocket? | DiffSBDD, TargetDiff, DiffGui |
| `common/` | — | the shared arm registry and significance tests | — |

## Environments

`common`, `ctbench` and `sbddbench` are members of the root uv workspace and
share the repository's `.venv`.

**`plbench` is not**, and this is deliberate: it runs ESM3 in-process, and ESM3
pins a fork of `transformers` that would replace the `transformers` 5.x the
ProLIT language models are built on. A uv workspace resolves one version per
package, so membership would silently downgrade the model library. `plbench`
keeps its own project and lockfile:

```sh
cd benchmarks/plbench && uv sync            # its own .venv
```

Its FoldToken and Bio2Token adapters go further still and run in dedicated
venvs (`.venv-foldtoken`, `.venv-bio2token`) as subprocesses, because those
pin exact CUDA-coupled torch builds.

The generative baselines under `sbddbench` are the same story in conda form:
each generates in its own environment and writes a plain `generated.sdf`, which
the benchmark environment then scores. See `sbddbench/README.md`.

## The arm registry

`common/src/prolit_bench/variants.py` defines which runs, normalization
statistics and codebook sizes each arm (`joint`, `separate`, `separate_4096`)
means. Read its module docstring before changing anything there — it documents a
real discrepancy between the tables that is recorded rather than hidden, and
`test_variant_agreement.py` fails if the benchmarks drift apart again.

## Running the analysis (no GPU)

```sh
uv run pytest benchmarks                    # metric, stats and reproduction tests
uv run python benchmarks/ctbench/scripts/make_tables.py
uv run python benchmarks/ctbench/scripts/make_figures.py
```

The reproduction tests assert that the analysis layer still produces the
published numbers from the committed per-sample dumps. One is an expected
failure: no committed TargetDiff dump reproduces the generation table's -4.76
(see `ctbench/tests/test_reproduce_generation.py`).

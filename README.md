# ProLIT — a structural tokenizer for 3D protein–ligand complexes

Protein–ligand complexes encode the interactions that matter in drug discovery,
but existing structure tokenizers represent proteins and ligands separately.
**ProLIT** (Protein–Ligand Interface Tokenizer) encodes pocket residues and
ligand atoms *together*, into one sequence of discrete symbols drawn from a
single chemistry-aware vocabulary.

A Transformer VQ-VAE maps the all-atom coordinates and chemistry of a complex to
those tokens and reconstructs its 3D structure from them. Two language models
are then trained directly on the token stream:

| model | what it is | what it does |
|---|---|---|
| **ProLIT-MLM** | encoder-only masked LM (~99 M) | ranks candidate binding poses |
| **ProLIT-CLM** | causal decoder (~298 M) + E(3)-equivariant flow-matching refiner | generates 3D ligands for a pocket |

Paper: *Learning the Language of the Binding Interface* (AAAI 2027 submission).

## Layout

```
src/prolit/        the library — tokenizer, models, datasets. Start at prolit/api.py.
pipelines/         corpus construction and training (CLIs)
benchmarks/        one per paper table; see benchmarks/README.md
scripts/           the ProLIT entry points the benchmarks drive as subprocesses
jobs/              cluster job submission (the scripts themselves stay local)
third_party/       baseline sources (submodules); patches/ holds our edits to them
notebooks/         marimo notebooks that produce the paper's figures
```

Dependencies only point one way:

```
pipelines/  ─┐
benchmarks/ ─┼──>  src/prolit/
scripts/    ─┘
```

`prolit` knows nothing about the layers above it, and those three are siblings
that never import each other — when two of them need the same thing, it belongs
in `prolit`. `tests/test_layering.py` enforces this, including the subprocess
edges and a cap on how many entry points `scripts/` may hold; it exists because
those edges are invisible in review, and every case it checks was a real
violation before it was written.

Only code is tracked. Trained weights, descriptor caches, token streams,
per-sample dumps, rendered figures and cluster job scripts are all git-ignored —
they are either outputs of the pipelines above or properties of one machine, and
they stay where they were produced. Notes and the frozen records of what each
run concluded live beside them under `docs/`, also untracked — this repository
carries the code that produces results, not the results or the writeups.

Nothing tracked here hardcodes a path on our cluster. Roots are derived from the
file's own location, external binaries (AutoDock Vina, Open Babel, ADFRsuite's
`prepare_receptor`) are resolved from `PATH` at call time, and every default can
be overridden with an environment variable.

## Setup

```sh
git submodule update --init --recursive
sh scripts/apply_patches.sh     # required local edits to third_party/
uv sync --all-packages          # library + in-workspace benchmarks
```

`benchmarks/recon-bench` has its own environment on purpose — it runs ESM3
in-process, and ESM3 pins a fork of `transformers` that would downgrade the one
the language models need. See `benchmarks/README.md`.

## Using the models

```python
import torch
from prolit.api import load_tokenizer, load_norm_stats, load_causal_lm

device = torch.device("cuda")
stats = load_norm_stats("data/descriptor_cache_allatom/normalization_stats.pt", device)
tokenizer = load_tokenizer(VQVAE_CKPT, codebook_size=8192, device=device, norm_stats=stats)
clm = load_causal_lm(LM_CKPT, codebook_size=8192, device=device)
```

`prolit/api.py` is the supported surface; anything outside its `__all__` is
internal. A checkpoint and its `normalization_stats.pt` must always travel
together — the wrong pairing produces plausible but mis-scaled coordinates
rather than an error.

Which weights each tokenizer arm means is defined once, in
`benchmarks/common/src/prolit_bench/variants.py`. Which checkpoint produced
which published number is recorded in `docs/results/` alongside the checkpoints
themselves, on the machine that trained them.

## Training

Stages run in order; each writes what the next reads. See `pipelines/README.md`.

```sh
# corpus
python pipelines/corpora/build_descriptors.py --out-dir data/descriptor_cache_allatom
python pipelines/corpora/tokenize_crossdocked.py --ckpt <vqvae> --out-dir data/lm_tokens_allatom

# models
python pipelines/train/vqvae.py    # the tokenizer
python pipelines/train/clm.py      # ProLIT-CLM
python pipelines/train/mlm.py      # ProLIT-MLM
python pipelines/train/head.py     # a pose-rescoring or affinity head
python pipelines/train/refiner.py  # the pose refiner
```

These are plain CLIs — run them however your site expects. On a scheduler,
`jobs/submit.py` renders the boilerplate around one, and prints the `qsub` line
rather than submitting it. See `jobs/README.md`.

## Reproducibility

Every entry point that draws random numbers takes `--seed` (default 0) and
`--deterministic`, and applies both before doing anything:

```sh
python pipelines/train/clm.py --seed 7
python scripts/generate_ligands_for_target.py --seed 7 ...
```

`--seed` makes a run repeatable on the same machine and library versions.
`--deterministic` additionally asks torch and cuDNN for deterministic kernels —
slower, and it raises on operations that have no deterministic implementation.

Everything goes through `prolit.seeding`: it seeds Python, NumPy and torch
(CPU + CUDA), gives DataLoader workers their own NumPy streams (torch seeds only
its own RNG in workers), and hands independent named streams to components that
need them, so two of them never draw from the same sequence. A training run
records its seed in the checkpoint's hyperparameters.

## Development

```sh
uv run pytest          # library, pipelines and in-workspace benchmarks
uv run ruff check .
uv run ty check src
```

Conventions: dataclass configs in `prolit/config.py` (no Hydra, despite what
older notes say), `LightningModule` / `LightningDataModule` for models and data,
Ruff with `select = ["ALL"]`, marimo notebooks as `.py`. Code and commit messages
in English.

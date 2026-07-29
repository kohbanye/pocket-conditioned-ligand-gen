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
jobs/              TSUBAME submission tooling + the archive of jobs already run
scripts/           evaluation and generation entry points
third_party/       baseline sources (submodules); patches/ holds our edits to them
docs/              frozen result records and dated investigation notes
notebooks/         marimo notebooks that produce the paper's figures
```

## Setup

```sh
git submodule update --init --recursive
sh scripts/apply_patches.sh     # required local edits to third_party/
uv sync --all-packages          # library + in-workspace benchmarks
```

`benchmarks/plbench` has its own environment on purpose — it runs ESM3
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

Which checkpoints correspond to which paper numbers is recorded in
`docs/results/best_allatom_configs.md`, and which weights each tokenizer arm
means is defined once in `benchmarks/common/src/prolit_bench/variants.py`.

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

On TSUBAME, wrap these with `jobs/submit.py` rather than hand-writing a job
script — it emits the prologue that avoids a known silent-failure mode.

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

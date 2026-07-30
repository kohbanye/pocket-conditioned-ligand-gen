"""One way to seed a run, and one way to spell the ``--seed`` flag.

Randomness enters this project in more places than is obvious: weight init and
codebook init, the MLM's dynamic masking, the pose refiner's online jitter,
rotation augmentation during tokenization, DataLoader shuffling and its worker
processes, nucleus sampling during generation, and the perturbations that build
decoy corpora. Seeding some of them is worse than seeding none, because the run
looks reproducible until it isn't.

:func:`seed_everything` covers the global generators; anything that needs its
own stream takes a seed explicitly (see :func:`derive_seed`), so two components
in one run do not silently draw from the same sequence.

Determinism has a cost. :func:`seed_everything` makes a run *repeatable* on the
same machine and library versions, which is what reproducing a number needs. It
does not force bit-identical GPU kernels -- pass ``deterministic=True`` for that,
and expect it to be slower and to fail loudly on ops that have no deterministic
implementation.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import argparse

#: Used when a caller does not pass one. Every entry point defaults to this, so
#: two runs of two different scripts start from the same place unless told
#: otherwise.
DEFAULT_SEED = 0


def seed_everything(seed: int = DEFAULT_SEED, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy and torch (CPU and CUDA). Returns the seed.

    ``deterministic`` additionally asks torch and cuDNN for deterministic
    kernels. That makes results bit-identical across runs on the same hardware
    at some cost in speed, and raises on operations with no deterministic
    implementation -- useful when chasing a discrepancy, too strict as a default.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002  (legacy global RNG; some deps still use it)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(mode=True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def derive_seed(seed: int, name: str) -> int:
    """A stable sub-seed for one named component of a run.

    Two things that need independent streams -- say the corpus shuffle and the
    pose jitter -- must not both start from ``seed``, or their draws correlate.
    Deriving by name keeps each stream reproducible and independent, and keeps
    the mapping stable across runs and machines (unlike ``hash()``, which is
    salted per process).
    """
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def rng_for(seed: int, name: str) -> np.random.Generator:
    """A NumPy generator for one named component, seeded via :func:`derive_seed`."""
    return np.random.default_rng(derive_seed(seed, name))


def torch_generator(seed: int, name: str) -> torch.Generator:
    """A torch generator for one named component (e.g. DataLoader shuffling)."""
    return torch.Generator().manual_seed(derive_seed(seed, name))


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own reproducible NumPy / Python stream.

    torch seeds each worker's torch RNG itself, but leaves NumPy and ``random``
    alone -- so without this every worker draws the *same* NumPy sequence, and a
    dataset that jitters or masks with NumPy produces duplicate augmentations
    across workers. Derived from the torch seed, so it still follows
    :func:`seed_everything`.
    """
    base = torch.initial_seed() % 2**32
    np.random.seed((base + worker_id) % 2**32)  # noqa: NPY002
    random.seed(base + worker_id)


def add_seed_argument(
    parser: argparse.ArgumentParser,
    *,
    default: int = DEFAULT_SEED,
) -> argparse.ArgumentParser:
    """Add the standard ``--seed`` / ``--deterministic`` pair to a CLI."""
    parser.add_argument(
        "--seed",
        type=int,
        default=default,
        help=f"random seed for this run (default: {default})",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="also force deterministic torch/cuDNN kernels; slower, and raises "
        "on ops with no deterministic implementation",
    )
    return parser


def seed_from_args(args: argparse.Namespace) -> int:
    """Seed the run from a parsed namespace carrying ``--seed``."""
    return seed_everything(
        getattr(args, "seed", DEFAULT_SEED),
        deterministic=getattr(args, "deterministic", False),
    )

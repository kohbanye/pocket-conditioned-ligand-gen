"""Seeding is only useful if it is total.

Seeding some generators and not others is worse than seeding none: the run looks
reproducible until the unseeded one changes an answer. So this checks both that
the helpers work and that every entry point with randomness in it actually takes
a ``--seed`` and applies it -- the second is the part that rots.
"""

from __future__ import annotations

import ast
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

from prolit.config import (
    AtomVQVAETrainingConfig,
    LMTrainingConfig,
    MLMTrainingConfig,
    PoseRefineTrainingConfig,
    RescoreTrainingConfig,
)
from prolit.seeding import (
    DEFAULT_SEED,
    derive_seed,
    rng_for,
    seed_everything,
    torch_generator,
    worker_init_fn,
)

REPO = Path(__file__).resolve().parent.parent

#: Markers that a file draws random numbers. Deliberately broad: a false
#: positive costs one ``--seed`` flag, a false negative costs reproducibility.
_RANDOM_MARKERS = (
    "default_rng",
    "np.random",
    "torch.rand",
    "torch.multinomial",
    "randperm",
    "random_rotation",
    "random.",
    "multinomial",
    "randomSeed",
)


def _entry_points() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "scripts", "pipelines", "benchmarks"],  # noqa: S607
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.split("\0")
    paths = [REPO / f for f in out if f.endswith(".py")]
    return [p for p in paths if "parse_args()" in p.read_text()]


def test_seed_everything_makes_draws_repeatable() -> None:
    def draws() -> tuple:
        return (
            random.random(),  # noqa: S311
            float(np.random.rand()),  # noqa: NPY002
            float(torch.rand(1)),
        )

    seed_everything(1234)
    first = draws()
    seed_everything(1234)
    assert draws() == first

    seed_everything(4321)
    assert draws() != first


def test_derived_streams_are_independent_and_stable() -> None:
    """Two named streams from one seed must not produce the same numbers."""
    a = rng_for(7, "jitter").normal(size=8)
    b = rng_for(7, "masking").normal(size=8)
    assert not np.allclose(a, b)
    # ...and must be reproducible across processes (not salted like hash()).
    assert derive_seed(7, "jitter") == derive_seed(7, "jitter")
    assert np.allclose(rng_for(7, "jitter").normal(size=8), a)


def test_torch_generator_is_seeded_per_name() -> None:
    g1 = torch_generator(3, "shuffle")
    g2 = torch_generator(3, "shuffle")
    assert torch.equal(
        torch.randperm(16, generator=g1), torch.randperm(16, generator=g2)
    )
    other = torch_generator(3, "jitter")
    assert not torch.equal(
        torch.randperm(16, generator=torch_generator(3, "shuffle")),
        torch.randperm(16, generator=other),
    )


def test_worker_init_gives_each_worker_its_own_stream() -> None:
    """Without this every DataLoader worker draws the same NumPy sequence."""
    seed_everything(5)
    worker_init_fn(0)
    first = np.random.rand(4)  # noqa: NPY002
    seed_everything(5)
    worker_init_fn(1)
    second = np.random.rand(4)  # noqa: NPY002
    assert not np.allclose(first, second)

    seed_everything(5)
    worker_init_fn(0)
    assert np.allclose(np.random.rand(4), first)  # noqa: NPY002


def test_every_random_entry_point_takes_a_seed() -> None:
    """A CLI that draws random numbers must expose --seed and apply it."""
    missing_flag, missing_call = [], []
    for path in _entry_points():
        text = path.read_text()
        if not any(m in text for m in _RANDOM_MARKERS):
            continue
        rel = str(path.relative_to(REPO))
        if "add_seed_argument" not in text and '"--seed"' not in text:
            missing_flag.append(rel)
        if "seed_from_args" not in text and "seed_everything" not in text:
            missing_call.append(rel)
    assert not missing_flag, "no --seed flag:\n  " + "\n  ".join(sorted(missing_flag))
    assert not missing_call, (
        "never seeds the run:\n  " + "\n  ".join(sorted(missing_call))
    )


def test_training_configs_record_their_seed() -> None:
    """The seed belongs in the checkpoint, so a run can say what produced it."""
    for cls in (
        AtomVQVAETrainingConfig,
        LMTrainingConfig,
        MLMTrainingConfig,
        RescoreTrainingConfig,
        PoseRefineTrainingConfig,
    ):
        assert cls().seed == DEFAULT_SEED, cls.__name__


def test_training_scripts_pass_the_seed_into_the_config() -> None:
    """Otherwise --seed would seed the globals but not be recorded anywhere."""
    offenders = [
        p.name
        for p in sorted((REPO / "pipelines" / "train").glob("*.py"))
        if "config.seed = args.seed" not in p.read_text()
    ]
    assert not offenders, f"do not record --seed in their config: {offenders}"


def test_no_dataset_creates_an_unseeded_generator() -> None:
    """``np.random.default_rng()`` with no argument is never reproducible."""
    offenders = []
    for path in (REPO / "src" / "prolit").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            call_is_default_rng = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "default_rng"
                and not node.args
                and not node.keywords
            )
            if call_is_default_rng:
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert not offenders, (
        "unseeded default_rng() in the library:\n  " + "\n  ".join(offenders)
    )

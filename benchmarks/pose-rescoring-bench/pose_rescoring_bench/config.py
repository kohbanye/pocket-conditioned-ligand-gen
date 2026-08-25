"""Dataclass configuration groups (stdlib ``@dataclass`` idiom, typed defaults).

Plain dataclasses with typed defaults, instantiated directly in the ``scripts/*``
entry points. Paths default to locations inside this monorepo and can be
overridden per run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# This file sits three levels below the monorepo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_GIT_ROOT = _REPO_ROOT.parent


def _source_repo_default() -> Path:
    """Model-library root. ``RESCORING_BENCH_SOURCE_REPO`` overrides it for
    portability."""
    env = os.environ.get("RESCORING_BENCH_SOURCE_REPO")
    return Path(env) if env else _REPO_ROOT


@dataclass
class PathsConfig:
    """Filesystem locations for models, data, baselines and results."""

    source_repo: Path = field(default_factory=_source_repo_default)
    # RTMScore / GenScore live in their own checkout with micromamba envs; they
    # are not vendored here because each pins a conflicting DGL build.
    baselines_repo: Path = _GIT_ROOT / "baselines"
    sbdd_bench_repo: Path = _REPO_ROOT / "benchmarks" / "sbdd-bench"
    results_dir: Path = Path("results")

    @property
    def casf_dir(self) -> Path:
        return self.source_repo / "data" / "casf2016"

    @property
    def targets_dir(self) -> Path:
        return self.source_repo / "data" / "targets"

    @property
    def norm_stats(self) -> Path:
        return (
            self.source_repo
            / "data"
            / "descriptor_cache_allatom"
            / "normalization_stats.pt"
        )

    @property
    def bench_python(self) -> Path:
        """Interpreter for the sbdd-bench (scoring) env.

        The bench and each model live in mutually incompatible venvs, so the
        crossdocked driver subprocesses this explicitly rather than relying on
        whichever ``python`` happens to be on ``PATH``. Override with
        ``RESCORING_BENCH_SBDD_PYTHON``.
        """
        env = os.environ.get("RESCORING_BENCH_SBDD_PYTHON")
        return Path(env) if env else self.sbdd_bench_repo / ".venv" / "bin" / "python"

    def ckpt(self, rel: str) -> Path:
        """Resolve a checkpoint path stated relative to the source repo."""
        return self.source_repo / rel


@dataclass
class GenerationConfig:
    """Pocket-conditioned generation inference settings."""

    targets: tuple[str, ...] = ("2ity", "1iep", "3pbl")
    n_samples: int = 150
    temperature: float = 0.85
    top_p: float = 1.0
    use_refiner: bool = True
    max_residues: int = 50
    seed: int = 42


@dataclass
class RescoringConfig:
    """CASF pose-rescoring inference settings."""

    score_mode: str = "head"  # pll | head | ensemble
    # Average the head score over this many random frame rotations. The
    # score of a pose should not depend on the arbitrary frame the pocket
    # extraction picked, but it does: across frames a pose's predicted RMSD
    # has a median spread of 0.32 A, while the score gap that decides which
    # pose gets picked is 0.28 A -- the frame noise is larger than the
    # signal. It falls as 1/sqrt(k), so averaging restores the invariance
    # the model was trained for but never guaranteed.
    n_frames: int = 1
    exclude_native: bool = True
    native_thresh: float = 2.0
    max_residues: int = 50
    max_targets: int | None = None


@dataclass
class AffinityConfig:
    """CASF affinity inference settings."""

    label_cap: float = 13.0
    max_residues: int = 50
    max_targets: int | None = None


@dataclass
class EvalConfig:
    """Top-level configuration for one evaluation run."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    codebook_size: int = 8192
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    rescoring: RescoringConfig = field(default_factory=RescoringConfig)
    affinity: AffinityConfig = field(default_factory=AffinityConfig)

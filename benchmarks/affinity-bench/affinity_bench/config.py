"""Dataclass configuration for one affinity evaluation run.

Plain dataclasses with typed defaults, instantiated in the ``scripts/*`` entry
points. Paths default to locations inside this monorepo and are overridable per
run so the bench is not pinned to one machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# benchmarks/affinity-bench/affinity_bench/config.py -> up three.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_GIT_ROOT = _REPO_ROOT.parent


def _source_repo_default() -> Path:
    """Model-library root; ``PROLIT_SOURCE_REPO`` overrides it for portability."""
    env = os.environ.get("PROLIT_SOURCE_REPO")
    return Path(env) if env else _REPO_ROOT


@dataclass
class PathsConfig:
    """Where the models, the benchmark data and the baselines live."""

    source_repo: Path = field(default_factory=_source_repo_default)
    # GenScore lives in its own checkout with a micromamba env; it is not
    # vendored here because it pins a conflicting DGL build.
    baselines_repo: Path = _GIT_ROOT / "baselines"
    results_dir: Path = Path("results")

    @property
    def casf_dir(self) -> Path:
        return self.source_repo / "data" / "casf2016"

    @property
    def norm_stats(self) -> Path:
        return (
            self.source_repo
            / "data"
            / "descriptor_cache_allatom"
            / "normalization_stats.pt"
        )

    def ckpt(self, rel: str) -> Path:
        """Resolve a checkpoint path stated relative to the source repo."""
        return self.source_repo / rel


@dataclass
class AffinityConfig:
    """How a complex is scored.

    ``n_frames`` averages the prediction over that many random rigid rotations
    of the whole complex. The pocket-canonical frame is derived from the pocket,
    so a complex's affinity should not depend on it -- but quantization makes it
    depend on it anyway, and the resulting spread is independent of the pK while
    the signal is not. Averaging therefore attenuates a noise term that
    otherwise pulls both Pearson R and the within-cluster Spearman toward zero.
    The pose head needed 16 draws before the spread stopped falling; the same
    value is used here rather than one tuned on CASF.
    """

    n_frames: int = 1
    max_residues: int = 50
    max_targets: int | None = None


@dataclass
class EvalConfig:
    """Top-level configuration for one evaluation run."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    affinity: AffinityConfig = field(default_factory=AffinityConfig)

"""Fetch the published checkpoints so a fresh machine can run without the cluster.

Weights are not in git (see CLAUDE.md); they live in a Hugging Face repo. This
module is the single place that knows which file is which, so a caller never has
to guess -- and in particular never has to pair a checkpoint with the wrong
normalization statistics.

**The tokenizer is a set.** ``atom_vqvae`` and ``norm_stats`` must come from the
same training run. Mixing them does not raise: it produces plausible coordinates
at the wrong scale, which is the most expensive failure in this repository
because nothing downstream complains. :func:`fetch_group` exists so the set is
requested by name rather than assembled by hand.

Every ``.ckpt`` here is a Lightning checkpoint, which pickles the *config
instance*; pickle records a class by (module path, class name). Renaming a
dataclass in :mod:`prolit.config` therefore makes older checkpoints unloadable.
The repo commit that produced these is recorded in the HF model card.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Default Hugging Face repo. Override with ``PROLIT_WEIGHTS_REPO``.
DEFAULT_REPO = "kohbanye/prolit-weights"

#: Logical name -> path inside the HF repo.
FILES: dict[str, str] = {
    "atom_vqvae": "tokenizer/atom_vqvae_e244_coord0.1022.ckpt",
    "norm_stats": "tokenizer/normalization_stats.pt",
    "code_neighbours": "tokenizer/code_neighbours_e250lig3.pt",
    "clm": "clm/lm-e00-vl5.3568.ckpt",
    "mlm": "mlm/mlm_iter3-e00-vl6.1401.ckpt",
    "refiner": "refiner/refit_press0.6.ckpt",
    "refiner_geom": "refiner/refit_geom_lb10.ckpt",
    "refiner_new": "refiner/refit_new_lb10.ckpt",
    "refiner_clm": "refiner/refclm.ckpt",
    "refiner_torsion": "refiner/refit_tors3_ts0.4-e10.ckpt",
    "refiner_torsion_only": "refiner/refit_torsonly_s01-e03.ckpt",
}

#: Named bundles. ``tokenizer`` is a set on purpose -- see the module docstring.
GROUPS: dict[str, tuple[str, ...]] = {
    "tokenizer": ("atom_vqvae", "norm_stats"),
    "generate": ("atom_vqvae", "norm_stats", "clm", "refiner"),
    "iterative": ("atom_vqvae", "norm_stats", "clm", "mlm", "code_neighbours"),
    "all": tuple(FILES),
}

#: Environment variables the sbdd-bench ``own`` adapter reads, by logical name.
ENV_FOR: dict[str, str] = {
    "atom_vqvae": "SBDD_OWN_VQVAE_CKPT",
    "norm_stats": "SBDD_OWN_NORM_STATS",
    "clm": "SBDD_OWN_LM_CKPT",
    "refiner": "SBDD_OWN_REFINE_CKPT",
    "mlm": "SBDD_OWN_MLM_CKPT",
}


def fetch(name: str, *, repo: str | None = None, token: str | None = None) -> Path:
    """Download one checkpoint and return its local path (cached across calls)."""
    if name not in FILES:
        msg = f"unknown weight {name!r}; known: {sorted(FILES)}"
        raise KeyError(msg)
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    return Path(
        hf_hub_download(
            repo_id=repo or os.environ.get("PROLIT_WEIGHTS_REPO", DEFAULT_REPO),
            filename=FILES[name],
            token=token,
        )
    )


def fetch_group(
    group: str = "generate", *, repo: str | None = None, token: str | None = None
) -> dict[str, Path]:
    """Download a named bundle and return ``{logical name: local path}``."""
    if group not in GROUPS:
        msg = f"unknown group {group!r}; known: {sorted(GROUPS)}"
        raise KeyError(msg)
    return {n: fetch(n, repo=repo, token=token) for n in GROUPS[group]}


def env_lines(paths: dict[str, Path]) -> list[str]:
    """``export VAR=path`` lines for the fetched files the benchmark reads."""
    return [
        f"export {ENV_FOR[name]}={path}"
        for name, path in paths.items()
        if name in ENV_FOR
    ]

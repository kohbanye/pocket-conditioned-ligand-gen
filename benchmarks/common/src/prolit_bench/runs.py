"""Resolve a training run's name to one checkpoint file.

``pipelines/train/*.py`` writes to ``<project>/<run-name>/checkpoints/``, where
``<project>`` is the wandb project the trainer logs to. A benchmark therefore
refers to a trained component either by an exact checkpoint path or by the run
name, and something has to turn the second into the first.

That policy lives here rather than in either benchmark because two of them need
it and they are siblings: the pose table and the affinity table select a head
out of the same directory tree, and a run-name that resolved to a different
epoch depending on which table you went through is exactly the class of silent
disagreement ``variants.py`` exists to prevent.

No torch: every benchmark depends on this package, including the one that
cannot share the main environment.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

#: Training project directories, keyed by the component they hold.
PROJECTS = {
    "scoring_head": "pocket-ligand-rescore",
    "mlm": "pocket-ligand-mlm",
    "clm": "pocket-ligand-lm",
    "vqvae": "pocket-ligand-vqvae",
    "refiner": "pocket-ligand-refine",
}

#: ``rescore-e09-vl0.6196.ckpt`` -> 0.6196
_SCORING_VAL_LOSS = re.compile(r"rescore-e\d+-vl([0-9.]+)\.ckpt$")


def scoring_head_val_loss(path: Path) -> float:
    """The val loss encoded in a scoring-head checkpoint's filename.

    Returns infinity for a name that does not carry one, so such a file loses
    every comparison rather than winning by parsing as zero.
    """
    match = _SCORING_VAL_LOSS.search(path.name)
    return float(match.group(1)) if match is not None else float("inf")


def resolve_scoring_head(source_repo: Path, spec: str) -> Path:
    """Resolve a scoring head: an exact ``*.ckpt`` path, or a run name to select in.

    A run name selects the LOWEST-val-loss checkpoint of that run. Selecting on
    val loss rather than on the last epoch is the choice being stated: the head
    is small and overfits a small corpus, so its last epoch is routinely not its
    best, and picking by CASF score instead would be selecting on the test set.
    """
    if spec.endswith(".ckpt"):
        return source_repo / spec
    ckpt_dir = source_repo / PROJECTS["scoring_head"] / spec / "checkpoints"
    candidates = sorted(ckpt_dir.glob("rescore-*.ckpt"))
    if not candidates:
        msg = (
            f"no scoring-head checkpoints for run {spec!r} under {ckpt_dir}. "
            "Still training, or the run name is wrong."
        )
        raise FileNotFoundError(msg)
    return min(candidates, key=scoring_head_val_loss)

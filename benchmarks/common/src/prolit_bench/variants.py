"""The single source of truth for which weights each tokenizer arm means.

Three benchmarks report on the same arms in three different paper tables. Until
this module existed they each resolved arms independently -- ``ctbench`` from a
hand-written registry, ``plbench`` from an ``ARMS`` dict inside its adapter --
and the two had drifted apart in a way no test could see: for the
separately-trained VQ-VAEs, ``ctbench`` pinned each run's ``last.ckpt`` while
``plbench`` picked the run's lowest-``val/atom_coord`` checkpoint. Those are not
the same weights (the separate runs' best epochs are 95-98, not 99), so the
reconstruction table and the rescoring/generation tables were describing
slightly different models under one name.

This module does NOT pick a winner. Both sets of published numbers were computed
under their own policy, and silently switching either would invalidate results
already in the paper without re-running anything. What it does instead:

* An arm's *identity* -- which runs, which normalization statistics, which
  codebook size -- is defined once, here, so the benchmarks cannot disagree
  about that.
* Checkpoint selection is a policy the caller states
  (:func:`checkpoints`\'s ``select``), and each benchmark declares the policy it
  published under in one visible place.
* ``benchmarks/test_variant_agreement.py`` asserts the identities match and
  reports, as a readable diff, exactly which files the two policies disagree on.

Resolving the inconsistency means re-running one of the tables; that is a call
for whoever owns the paper, not something to bury in a refactor.

Run directories are named, not paths: a run is a directory under
``pocket-ligand-vqvae/``, and :func:`resolve_vqvae` turns (run, policy) into a
file. Normalization statistics always travel with a checkpoint -- the wrong
pairing yields plausible but mis-scaled coordinates rather than an error -- so
they are part of the arm, not a separate argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# The monorepo root: benchmarks/common/src/prolit_bench/variants.py -> up four.
REPO_ROOT = Path(__file__).resolve().parents[4]
VQ_RUNS_DIR = REPO_ROOT / "pocket-ligand-vqvae"
ALLATOM_CACHE = REPO_ROOT / "data" / "descriptor_cache_allatom"

SelectPolicy = Literal["best", "last"]


@dataclass(frozen=True)
class Arm:
    """One tokenizer configuration, and exactly which files it resolves to."""

    name: str
    label: str
    protein_run: str
    ligand_run: str
    protein_norm: Path
    ligand_norm: Path
    #: Per-modality codebook size. The combined code space is twice this for a
    #: separate arm, and equal to it for the joint arm (one shared book).
    codebook_size: int
    #: Whether the two runs are the same model (joint) or two (separate).
    is_separate: bool
    #: An explicit checkpoint filename inside the run, when the arm is pinned to
    #: one rather than selected by policy. The joint arm is pinned -- both
    #: benchmarks already named the same file -- and its run has no
    #: ``last.ckpt``, so no policy applies to it.
    pinned_ckpt: str | None = None
    notes: str = ""

    @property
    def combined_codebook_size(self) -> int:
        """Size of the code space the language models see."""
        return 2 * self.codebook_size if self.is_separate else self.codebook_size


def resolve_vqvae(
    run: str,
    select: SelectPolicy,
    min_epoch: int = 90,
    pinned: str | None = None,
) -> Path:
    """Resolve a run name + policy to one checkpoint file.

    ``pinned`` short-circuits the policy: an arm may name an exact checkpoint,
    in which case that is what every task uses.

    ``"last"`` takes the run's ``last.ckpt``. ``"best"`` takes the lowest
    ``val/atom_coord`` checkpoint at or past ``min_epoch``; the epoch floor
    refuses a half-trained run, which would otherwise become a quiet row in an
    ablation table.

    The monitored metric contains a ``/``, so Lightning writes each checkpoint
    into its own directory: ``<run>/checkpoints/atomvqvae-epoch=NN-val/atom_coord=X.ckpt``.
    """
    ckpt_dir = VQ_RUNS_DIR / run / "checkpoints"
    if pinned is not None:
        path = ckpt_dir / pinned
        if not path.exists():
            msg = f"run {run!r}: pinned checkpoint missing: {path}"
            raise FileNotFoundError(msg)
        return path
    if select == "last":
        path = ckpt_dir / "last.ckpt"
        if not path.exists():
            msg = f"run {run!r}: no last.ckpt in {ckpt_dir}"
            raise FileNotFoundError(msg)
        return path

    found = []
    for path in ckpt_dir.glob("*/atom_coord=*.ckpt"):
        epoch = int(path.parent.name.split("epoch=")[1].split("-")[0])
        if epoch >= min_epoch:
            found.append((float(path.stem.split("=")[-1]), epoch, path))
    if not found:
        msg = (
            f"run {run!r}: no checkpoint past epoch {min_epoch} in {ckpt_dir}. "
            "Still training, or the run name is wrong."
        )
        raise FileNotFoundError(msg)
    return min(found)[2]


JOINT = Arm(
    name="joint",
    label="ProLIT (one shared codebook)",
    protein_run="xzkjxu9q",
    ligand_run="xzkjxu9q",
    protein_norm=ALLATOM_CACHE / "normalization_stats.pt",
    ligand_norm=ALLATOM_CACHE / "normalization_stats.pt",
    codebook_size=8192,
    is_separate=False,
    pinned_ckpt="atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt",
    notes="one 8192 book covering pocket atoms and ligand atoms alike",
)

SEPARATE = Arm(
    name="separate",
    label="Separate 8192+8192 (rate-matched)",
    protein_run="protein-vqvae",
    ligand_run="ligand-vqvae",
    protein_norm=ALLATOM_CACHE / "normalization_stats_protein.pt",
    ligand_norm=ALLATOM_CACHE / "normalization_stats_ligand.pt",
    codebook_size=8192,
    is_separate=True,
    notes="same bits/atom as joint, twice the codebook vectors",
)

SEPARATE_4096 = Arm(
    name="separate_4096",
    label="Separate 4096+4096 (capacity-matched)",
    protein_run="protein-vqvae-4096",
    ligand_run="ligand-vqvae-4096",
    protein_norm=ALLATOM_CACHE / "normalization_stats_protein.pt",
    ligand_norm=ALLATOM_CACHE / "normalization_stats_ligand.pt",
    codebook_size=4096,
    is_separate=True,
    notes="same codebook vectors and LM vocabulary as joint, 12 bits/atom",
)

REGISTRY: dict[str, Arm] = {a.name: a for a in (JOINT, SEPARATE, SEPARATE_4096)}

#: Aliases the benchmarks grew independently for the same arm.
ALIASES: dict[str, str] = {"separate4096": "separate_4096"}


def get(name: str) -> Arm:
    """Look up an arm by name or by a benchmark's historical alias."""
    key = ALIASES.get(name, name)
    if key not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        msg = f"unknown tokenizer arm {name!r}; known: {known}"
        raise KeyError(msg)
    return REGISTRY[key]


#: The checkpoint-selection policy each benchmark's published numbers used.
#: These differ for the separate arms, which is the inconsistency described at
#: the top of this module. Change one only together with re-running its table.
PUBLISHED_POLICY: dict[str, SelectPolicy] = {
    "reconstruction": "best",  # plbench, paper Table 1
    "rescoring": "last",  # ctbench, paper Table 2
    "generation": "last",  # ctbench + sbddbench, paper Table 3
}


def checkpoints(
    name: str,
    select: SelectPolicy,
    min_epoch: int = 90,
) -> dict[str, Path]:
    """Resolve an arm plus a selection policy to concrete files.

    ``select`` is required rather than defaulted: which checkpoint of a run you
    take is a reportable choice, so a caller has to state it. Pass
    ``PUBLISHED_POLICY[task]`` to reproduce what a paper table used.
    """
    arm = get(name)
    return {
        "protein_ckpt": resolve_vqvae(
            arm.protein_run, select, min_epoch, arm.pinned_ckpt
        ),
        "ligand_ckpt": resolve_vqvae(
            arm.ligand_run, select, min_epoch, arm.pinned_ckpt
        ),
        "protein_norm": arm.protein_norm,
        "ligand_norm": arm.ligand_norm,
    }

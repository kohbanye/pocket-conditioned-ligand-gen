"""Tokenizer-variant registry for the ablation study.

The ablation axis is *how the tokenizer was trained*: a jointly-trained
protein-ligand COMPLEX tokenizer vs SEPARATELY-trained protein-only + ligand-only
tokenizers stitched into one combined code space, each carried through the full
downstream pipeline of every task while the downstream protocol is held identical.

Four variants are registered. ``joint`` is the paper baseline (single combined
VQ, ``codebook_size`` 8192). ``joint_nocasf`` is the fair joint-side control that
shares the exact downstream protocol (MLM + heads) trained alongside the separate
arm, so the two ablation arms differ *only* in the tokenizer. ``separate`` loads
two single-modality VQ-VAEs (protein codes ``[0, 8192)``, ligand ``[8192, 16384)``)
via :class:`src.tokenizers.separate_vqvae.SeparateVQVAE`, so its combined
``codebook_size`` is 16384. ``separate_4096`` is the FAIR-ablation redo of the
separate arm with 4096+4096 sub-codebooks (combined ``codebook_size`` 8192), so
its LM vocabulary matches the joint arm exactly (only pose + generation tasks).
Checkpoint strings are stated relative to the source repo and resolved with
``PathsConfig.ckpt`` (heads may be run-names, resolved by
:func:`ctbench.inference.encode.resolve_rescore_ckpt`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeadSpec:
    """One rescoring/affinity head: checkpoint + the pooling it was trained with.

    ``ckpt`` is either an exact path ending in ``.ckpt`` or a run-name (e.g.
    ``pose_head_sep``) resolved to the lowest-val-loss checkpoint of that run by
    :func:`ctbench.inference.encode.resolve_rescore_ckpt`.
    """

    ckpt: str
    pooling: str = "mean"
    label: str = ""


@dataclass(frozen=True)
class GenerationCkpts:
    """Checkpoints for the generation pipeline of one variant.

    A ``joint`` variant sets ``vqvae`` (single combined all-atom VQ) and the
    generator runs its ``--all-atom`` path. A ``separate`` variant leaves
    ``vqvae`` ``None`` and sets the ``protein_*``/``ligand_*`` pair instead; the
    generator then encodes the pocket with the protein-only VQ and decodes ligand
    codes with the ligand-only VQ over one combined code space (SeparateVQVAE).
    ``codebook_size`` is the PER-MODALITY size passed to the generator via
    ``--codebook-size`` (the generator doubles it for the separate combined space).
    """

    vqvae: str | None
    lm: str | None
    refiner: str | None = None
    codebook_size: int = 8192
    protein_vqvae: str | None = None
    ligand_vqvae: str | None = None
    protein_norm: str | None = None
    ligand_norm: str | None = None

    @property
    def is_separate(self) -> bool:
        """True for the separate-tokenizer arm (protein + ligand VQ-VAEs)."""
        return self.protein_vqvae is not None


@dataclass(frozen=True)
class RescoringCkpts:
    """Checkpoints for the pose-rescoring pipeline of one variant.

    A ``joint`` variant sets ``vqvae`` (single combined VQ). A ``separate`` variant
    leaves ``vqvae`` ``None`` and instead sets the ``protein_*``/``ligand_*`` pair
    (loaded into one combined code space); ``codebook_size`` is then the combined
    size (16384) while each sub-VQ uses half of it.
    """

    vqvae: str | None
    mlm: str | None
    heads: tuple[HeadSpec, ...] = ()
    codebook_size: int = 8192
    protein_vqvae: str | None = None
    ligand_vqvae: str | None = None
    protein_norm: str | None = None
    ligand_norm: str | None = None

    @property
    def is_separate(self) -> bool:
        """True for the separate-tokenizer arm (protein + ligand VQ-VAEs)."""
        return self.protein_vqvae is not None


@dataclass(frozen=True)
class AffinityCkpts:
    """Checkpoints for the affinity pipeline of one variant (ensemble of heads).

    Shares the joint/separate tokenizer convention of :class:`RescoringCkpts`.
    """

    vqvae: str | None
    mlm: str | None
    heads: tuple[HeadSpec, ...] = ()
    codebook_size: int = 8192
    protein_vqvae: str | None = None
    ligand_vqvae: str | None = None
    protein_norm: str | None = None
    ligand_norm: str | None = None

    @property
    def is_separate(self) -> bool:
        """True for the separate-tokenizer arm (protein + ligand VQ-VAEs)."""
        return self.protein_vqvae is not None


@dataclass(frozen=True)
class Variant:
    """A tokenizer variant and its per-task downstream checkpoints."""

    name: str
    description: str
    generation: GenerationCkpts | None = None
    rescoring: RescoringCkpts | None = None
    affinity: AffinityCkpts | None = None

    @property
    def trained(self) -> bool:
        """True once at least one task pipeline has real (non-None) checkpoints."""
        for task in (self.generation, self.rescoring, self.affinity):
            if task is None:
                continue
            if getattr(task, "vqvae", None) is not None:
                return True
            if getattr(task, "is_separate", False):
                return True
        return False


# jointly-trained all-atom complex tokenizer (paper baseline)
_JOINT_VQVAE = "pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"  # noqa: E501

# separately-trained single-modality VQ-VAEs + their per-modality RAW-descriptor
# normalization stats (combined code space assembled by SeparateVQVAE)
_PROTEIN_VQVAE = "pocket-ligand-vqvae/protein-vqvae/checkpoints/last.ckpt"
_LIGAND_VQVAE = "pocket-ligand-vqvae/ligand-vqvae/checkpoints/last.ckpt"
_PROTEIN_NORM = "data/descriptor_cache_allatom/normalization_stats_protein.pt"
_LIGAND_NORM = "data/descriptor_cache_allatom/normalization_stats_ligand.pt"

# FAIR-ablation redo: separate arm with 4096+4096 sub-codebooks (combined 8192).
# Norm stats are codebook-independent, so the 8192-arm files above are reused.
_PROTEIN_VQVAE_4096 = "pocket-ligand-vqvae/protein-vqvae-4096/checkpoints/last.ckpt"
_LIGAND_VQVAE_4096 = "pocket-ligand-vqvae/ligand-vqvae-4096/checkpoints/last.ckpt"

# MLM shared by the joint-side control (its non-CASF-leaking training run)
_NOCASF_MLM = "pocket-ligand-mlm/wxlhgqx3/checkpoints/mlm-e02-vl0.8199.ckpt"

# Pose refiner shared by every generation arm: it operates on generated 3D atoms
# and bonds (tokenizer-agnostic), so both ablation arms use the same checkpoint.
_REFINER = "pocket-ligand-refine/refine_atom_bond_v1/checkpoints/refine-e08-r0.9440.ckpt"  # noqa: E501

# Generation LMs retrained on the SAME enlarged all-poses cache (4.04B tokens) for
# the fair generation ablation: the joint-tokenizer control and the separate arm
# differ only in the tokenizer, both matched to identical training corpora. The
# paper-baseline generation model (JOINT below, p6lpk7br) is a separate deliverable.
_GEN_LM_JOINTNOCASF = "pocket-ligand-lm/lm_placement_joint2/checkpoints/last.ckpt"
_GEN_LM_SEPARATE = "pocket-ligand-lm/lm_placement_sep/checkpoints/last.ckpt"
_GEN_LM_SEPARATE_4096 = "pocket-ligand-lm/lm_placement_sep4096/checkpoints/last.ckpt"

JOINT = Variant(
    name="joint",
    description="Jointly-trained protein-ligand complex tokenizer (paper baseline).",
    generation=GenerationCkpts(
        vqvae=_JOINT_VQVAE,
        lm="pocket-ligand-lm/p6lpk7br/checkpoints/lm-e02-vl1.0029.ckpt",
        refiner="pocket-ligand-refine/refine_atom_bond_v1/checkpoints/refine-e08-r0.9440.ckpt",
    ),
    rescoring=RescoringCkpts(
        vqvae=_JOINT_VQVAE,
        mlm="pocket-ligand-mlm/j90rlrgm/checkpoints/mlm-e01-vl0.8528.ckpt",
        heads=(
            HeadSpec(
                "pocket-ligand-rescore/kdy9d8g3/checkpoints/rescore-e02-vl0.2075.ckpt",
                "mean",
                "v2",
            ),
        ),
    ),
    affinity=AffinityCkpts(
        vqvae=_JOINT_VQVAE,
        mlm="pocket-ligand-mlm/wxlhgqx3/checkpoints/mlm-e02-vl0.8199.ckpt",
        heads=(
            HeadSpec(
                "pocket-ligand-rescore/tzqaubl4/checkpoints/rescore-e09-vl0.6196.ckpt",
                "mean",
                "kdki-mean",
            ),
        ),
    ),
)

JOINT_NOCASF = Variant(
    name="joint_nocasf",
    description="Joint tokenizer with the separate arm's non-CASF downstream protocol.",
    generation=GenerationCkpts(
        vqvae=_JOINT_VQVAE,
        lm=_GEN_LM_JOINTNOCASF,
        refiner=_REFINER,
    ),
    rescoring=RescoringCkpts(
        vqvae=_JOINT_VQVAE,
        mlm=_NOCASF_MLM,
        heads=(HeadSpec("pose_head_jointnocasf", "mean", "v2"),),
        codebook_size=8192,
    ),
    affinity=AffinityCkpts(
        vqvae=_JOINT_VQVAE,
        mlm=_NOCASF_MLM,
        heads=(HeadSpec("aff_head_jointnocasf", "mean", "kdki"),),
        codebook_size=8192,
    ),
)

SEPARATE = Variant(
    name="separate",
    description="Separately-trained protein + ligand tokenizers in one code space.",
    generation=GenerationCkpts(
        vqvae=None,
        lm=_GEN_LM_SEPARATE,
        refiner=_REFINER,
        protein_vqvae=_PROTEIN_VQVAE,
        ligand_vqvae=_LIGAND_VQVAE,
        protein_norm=_PROTEIN_NORM,
        ligand_norm=_LIGAND_NORM,
    ),
    rescoring=RescoringCkpts(
        vqvae=None,
        protein_vqvae=_PROTEIN_VQVAE,
        ligand_vqvae=_LIGAND_VQVAE,
        protein_norm=_PROTEIN_NORM,
        ligand_norm=_LIGAND_NORM,
        mlm="pocket-ligand-mlm/mlm_nocasf_sep/checkpoints/last.ckpt",
        heads=(HeadSpec("pose_head_sep", "mean", "v2"),),
        codebook_size=16384,
    ),
    affinity=AffinityCkpts(
        vqvae=None,
        protein_vqvae=_PROTEIN_VQVAE,
        ligand_vqvae=_LIGAND_VQVAE,
        protein_norm=_PROTEIN_NORM,
        ligand_norm=_LIGAND_NORM,
        mlm="pocket-ligand-mlm/mlm_nocasf_sep/checkpoints/last.ckpt",
        heads=(HeadSpec("aff_head_sep", "mean", "kdki"),),
        codebook_size=16384,
    ),
)

SEPARATE_4096 = Variant(
    name="separate_4096",
    description="Separate 4096+4096 tokenizers matched to the joint 8192 LM vocab.",
    generation=GenerationCkpts(
        vqvae=None,
        lm=_GEN_LM_SEPARATE_4096,
        refiner=_REFINER,
        # PER-MODALITY size; the generator doubles it to the 8192 combined space.
        codebook_size=4096,
        protein_vqvae=_PROTEIN_VQVAE_4096,
        ligand_vqvae=_LIGAND_VQVAE_4096,
        protein_norm=_PROTEIN_NORM,
        ligand_norm=_LIGAND_NORM,
    ),
    rescoring=RescoringCkpts(
        vqvae=None,
        protein_vqvae=_PROTEIN_VQVAE_4096,
        ligand_vqvae=_LIGAND_VQVAE_4096,
        protein_norm=_PROTEIN_NORM,
        ligand_norm=_LIGAND_NORM,
        mlm="pocket-ligand-mlm/mlm_nocasf_sep4096/checkpoints/last.ckpt",
        heads=(HeadSpec("pose_head_sep4096", "mean", "v2"),),
        # COMBINED size (2*4096); each sub-VQ uses half.
        codebook_size=8192,
    ),
)

REGISTRY: dict[str, Variant] = {
    v.name: v for v in (JOINT, JOINT_NOCASF, SEPARATE, SEPARATE_4096)
}

ABLATION_ORDER: tuple[str, ...] = ("joint_nocasf", "separate", "separate_4096")


def get(name: str) -> Variant:
    """Look up a variant by name, raising a clear error for unknown names."""
    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        msg = f"unknown variant {name!r}; known: {known}"
        raise KeyError(msg)
    return REGISTRY[name]

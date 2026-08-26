"""Tokenizer-variant registry for the ablation study.

The ablation axis is *how the tokenizer was trained*: a jointly-trained
protein-ligand COMPLEX tokenizer vs SEPARATELY-trained protein-only + ligand-only
tokenizers stitched into one combined code space, each carried through the full
downstream pipeline of every task while the downstream protocol is held identical.

Three variants are registered. ``joint`` is the paper baseline (single combined
VQ, ``codebook_size`` 8192). ``joint_nocasf`` is the fair joint-side control that
shares the exact downstream protocol (MLM + heads) trained alongside the separate
arm, so the two ablation arms differ *only* in the tokenizer. ``separate`` loads
two single-modality VQ-VAEs of 4096 codes each (protein ``[0, 4096)``, ligand
``[4096, 8192)``) via :class:`prolit.tokenizers.separate_vqvae.SeparateVQVAE`, so
its combined ``codebook_size`` is 8192 and its LM vocabulary matches the joint
arm exactly -- the arm the paper describes.

An 8192+8192 separate arm was registered here too, before the ablation was
redone at matched total size. It is gone: its combined vocabulary was twice the
joint arm's, so it varied the token budget as well as the tokenizer, and the
paper reports the matched one. Its affinity head (``aff_head_sep``) had no
matched counterpart, so the separate arm now covers pose and generation only.
Checkpoint strings are stated relative to the source repo and resolved with
``PathsConfig.ckpt`` (heads may be run-names, resolved by
:func:`pose_rescoring_bench.inference.encode.resolve_rescore_ckpt`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeadSpec:
    """One rescoring/affinity head: checkpoint + the pooling it was trained with.

    ``ckpt`` is either an exact path ending in ``.ckpt`` or a run-name (e.g.
    ``pose_head_sep``) resolved to the lowest-val-loss checkpoint of that run by
    :func:`pose_rescoring_bench.inference.encode.resolve_rescore_ckpt`.
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

# Separately-trained single-modality VQ-VAEs, 4096 codes each so the combined
# space matches the joint arm's 8192 -- what the paper describes -- plus their
# per-modality RAW-descriptor normalization stats. The combined code space is
# assembled by SeparateVQVAE; the stats are codebook-independent.
_PROTEIN_VQVAE = "pocket-ligand-vqvae/protein-vqvae-4096/checkpoints/last.ckpt"
_LIGAND_VQVAE = "pocket-ligand-vqvae/ligand-vqvae-4096/checkpoints/last.ckpt"
_PROTEIN_NORM = "data/descriptor_cache_allatom/normalization_stats_protein.pt"
_LIGAND_NORM = "data/descriptor_cache_allatom/normalization_stats_ligand.pt"

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
_GEN_LM_SEPARATE = "pocket-ligand-lm/lm_placement_sep4096/checkpoints/last.ckpt"

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
    description="Separately-trained protein + ligand tokenizers, 4096 each.",
    generation=GenerationCkpts(
        vqvae=None,
        lm=_GEN_LM_SEPARATE,
        refiner=_REFINER,
        # PER-MODALITY size; the generator doubles it to the 8192 combined space.
        codebook_size=4096,
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
        mlm="pocket-ligand-mlm/mlm_nocasf_sep4096/checkpoints/last.ckpt",
        heads=(HeadSpec("pose_head_sep4096", "mean", "v2"),),
        # COMBINED size (2*4096); each sub-VQ uses half.
        codebook_size=8192,
    ),
)

# The rebuilt rescoring stack on the leak-free vq_e250_lig3 tokenizer. One head,
# no ensemble: the published "3-head" row averaged three heads picked by their
# CASF score, which is selection on the test set and cannot go in a paper. Here
# the pooling is chosen by DECOY VALIDATION LOSS and CASF is scored once.
#
# The decoy corpus is rebuilt too. tokenize_decoys already drops CASF and
# CrossDocked-test PDB ids, but the v16-family corpora also took their
# heavy-atom stratification FROM CASF, so the design itself leaked; these use
# the v10-style parameters with no stratification.
_E250_VQ = (
    "pocket-ligand-vqvae/vq_e250_lig3/checkpoints/"
    "atomvqvae-epoch=237-val/atom_coord=0.1021.ckpt"
)
_E250_MLM = "pocket-ligand-mlm/mlm_e250lig3/checkpoints/mlm-e02-vl0.8009.ckpt"

E250_MEAN = Variant(
    name="e250_mean",
    description="vq_e250_lig3 + leak-free MLM + a single mean-pooled head.",
    generation=GenerationCkpts(vqvae=_E250_VQ, lm=None, refiner=None),
    rescoring=RescoringCkpts(
        vqvae=_E250_VQ,
        mlm=_E250_MLM,
        heads=(HeadSpec("head_mean_e250lig3", "mean", "mean"),),
        codebook_size=8192,
    ),
    affinity=AffinityCkpts(vqvae=_E250_VQ, mlm=_E250_MLM, heads=(), codebook_size=8192),
)

E250_PAIRSUM = Variant(
    name="e250_pairsum",
    description="vq_e250_lig3 + leak-free MLM + a single pairwise-interaction head.",
    generation=GenerationCkpts(vqvae=_E250_VQ, lm=None, refiner=None),
    rescoring=RescoringCkpts(
        vqvae=_E250_VQ,
        mlm=_E250_MLM,
        heads=(HeadSpec("head_pairsum_e250lig3", "pairsum", "pairsum"),),
        codebook_size=8192,
    ),
    affinity=AffinityCkpts(vqvae=_E250_VQ, mlm=_E250_MLM, heads=(), codebook_size=8192),
)


#: The two arms the 2026-08-19 loop selected, both a SINGLE mean-pooled head --
#: no ensemble, no PLL term, no pairwise readout. What changed is the loss and,
#: for the second one, the corpus's ligand-size coverage.
#:
#: The published heads were trained with plain smooth-L1 regression on RMSD.
#: That optimises calibration everywhere, but docking power reads only the
#: argmax over a complex's poses: 97.9% of CASF targets already have a sub-2 A
#: pose in the head's top 5 (94.0% in the top 2 -- exactly RTMScore's DP@2A),
#: so every point of headroom sits in the last swap, not in the tail the
#: regression spends its capacity on. Adding the ListNet term over each
#: complex's poses -- weight 1.0, temperature 0.5, both already implemented and
#: both defaulting to off -- is worth +3.6 points of val DP@1A, consistent
#: across two seeds and two training lengths.
#:
#: ``_BIG`` additionally trains on a corpus built with ``--max-heavy 80``
#: instead of the default 50. The cap was arbitrary and it is exactly where
#: CASF falls off a cliff (DP@2A 89.5 for <=45 heavy atoms, 42.9 for 46-50,
#: 0.0 for >50). Sampling, seed and ``--n-complexes`` match the original build,
#: so the supplement is precisely the complexes that build discarded -- the
#: size histogram is NOT matched to CASF's, which would repeat the v16 mistake
#: named above. It narrows the val cliff from 14.4 to 8.4 points and costs 0.8
#: points on the <=45 tier, which is why both arms are kept and both reported.
_E250_VQ_BIG = _E250_VQ

E250_LISTWISE = Variant(
    name="e250_listwise",
    description=(
        "vq_e250_lig3 + leak-free MLM + a single mean-pooled head trained with "
        "smooth-L1 + ListNet over each complex's poses."
    ),
    generation=GenerationCkpts(vqvae=_E250_VQ, lm=None, refiner=None),
    rescoring=RescoringCkpts(
        vqvae=_E250_VQ,
        mlm=_E250_MLM,
        heads=(HeadSpec("head_e10_lw1.0_s7", "mean", "listwise"),),
        codebook_size=8192,
    ),
    affinity=AffinityCkpts(vqvae=_E250_VQ, mlm=_E250_MLM, heads=(), codebook_size=8192),
)

E250_LISTWISE_BIG = Variant(
    name="e250_listwise_big",
    description=(
        "As e250_listwise, but the decoy corpus keeps ligands up to 80 heavy "
        "atoms instead of 50 (size-coverage ablation)."
    ),
    generation=GenerationCkpts(vqvae=_E250_VQ_BIG, lm=None, refiner=None),
    rescoring=RescoringCkpts(
        vqvae=_E250_VQ_BIG,
        mlm=_E250_MLM,
        heads=(HeadSpec("head_big_lw1.0_s7", "mean", "listwise_big"),),
        codebook_size=8192,
    ),
    affinity=AffinityCkpts(
        vqvae=_E250_VQ_BIG, mlm=_E250_MLM, heads=(), codebook_size=8192
    ),
)


#: Generation on the retokenised stack. Shares ``_E250_VQ`` with the rescoring
#: arms rather than naming the file again -- one tokenizer, one string.
#:
#: The point of this arm is PoseBusters validity. Generation sat at 0.216 on the
#: old tokenizer, whose own CASP16 reconstruction is valid only 0.128 of the
#: time; this tokenizer reconstructs 0.668, so the ceiling the LM was pressed
#: against moves by 5.2x. Its corpora also drop the 54 generation-benchmark
#: pockets that ProLIT's fold split labelled train -- half the evaluation set
#: was inside the old CLM's training data.
#:
#: ``refiner=None`` on purpose: every refiner in the repo was measured less
#: accurate (best r 0.303) than this tokenizer's own ligand error (Kabsch
#: 0.329), so one can only add noise to what it is meant to repair.
E250_GEN = Variant(
    name="e250_gen",
    description="vq_e250_lig3 + CLM retrained on its tokens, leak-free corpora.",
    generation=GenerationCkpts(
        vqvae=_E250_VQ,
        lm="pocket-ligand-lm/clm_e250lig3_fullft/checkpoints/lm-e00-vl5.3568.ckpt",
        refiner=None,
        codebook_size=8192,
    ),
)


REGISTRY: dict[str, Variant] = {
    v.name: v
    for v in (
        JOINT,
        JOINT_NOCASF,
        SEPARATE,
        E250_MEAN,
        E250_PAIRSUM,
        E250_LISTWISE,
        E250_LISTWISE_BIG,
        E250_GEN,
    )
}

ABLATION_ORDER: tuple[str, ...] = ("joint_nocasf", "separate")


def get(name: str) -> Variant:
    """Look up a variant by name, raising a clear error for unknown names."""
    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        msg = f"unknown variant {name!r}; known: {known}"
        raise KeyError(msg)
    return REGISTRY[name]

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
    """One rescoring/affinity head: a checkpoint and the label its dump is named.

    ``ckpt`` is either an exact path ending in ``.ckpt`` or a run-name (e.g.
    ``pose_head_sep``) resolved to the lowest-val-loss checkpoint of that run by
    :func:`pose_rescoring_bench.inference.encode.resolve_rescore_ckpt`.
    """

    ckpt: str
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
        heads=(HeadSpec("pose_head_jointnocasf", "v2"),),
        codebook_size=8192,
    ),
    affinity=AffinityCkpts(
        vqvae=_JOINT_VQVAE,
        mlm=_NOCASF_MLM,
        heads=(HeadSpec("aff_head_jointnocasf", "kdki"),),
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
        heads=(HeadSpec("pose_head_sep4096", "v2"),),
        # COMBINED size (2*4096); each sub-VQ uses half.
        codebook_size=8192,
    ),
)


#: The tokenizer the paper reports: the joint all-atom VQ retrained to epoch
#: 237 (val atom_coord 0.1021 against the earlier 0.1073), paired with an MLM
#: pretrained on a corpus with the CASF-2016 core held out.
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
        heads=(HeadSpec("head_mean_e250lig3", "mean"),),
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
        heads=(HeadSpec("head_e10_lw1.0_s7", "listwise"),),
        codebook_size=8192,
    ),
    affinity=AffinityCkpts(vqvae=_E250_VQ, mlm=_E250_MLM, heads=(), codebook_size=8192),
)


#: :data:`E250_LISTWISE_LABEL` retrained on a decoy corpus that a docking
#: program could have produced.
#:
#: Two changes to ``tokenize_decoys.py``, nothing else. Every decoy is slid to
#: the least-clashing nearby offset, because CASF's own decoys are clash-free at
#: every RMSD band (0.083 clashes per heavy atom at 3-6 A) while ``_perturb``
#: reached 1.622 there; and six decoys per complex turn the ligand in place by
#: 60-180 deg, a pose the old corpus could not contain at all -- ``_perturb``
#: always rides a translation of up to 6 A on its rotation.
#:
#: The hole was measurable in the trained head. Rotating a CASF native in place
#: and scoring over 8 frames, predicted RMSD climbed only to ~60 deg and then
#: FELL: a flipped ligand at a true 3.72 A scored 2.03 A, better than a 45 deg
#: turn at 1.88 A. Pure translation tracked the truth to 5.89 against 5.90 over
#: the same range, so the failure was specific to orientation. It matches what
#: the DP@2A failures look like -- their top-scored pose sits 42.9 deg off the
#: native's principal axes against 7.3 deg for the targets that succeed.
#:
#: After retraining, on rotation axes never used to build the corpus (training
#: turns about the ligand's long axis; this test uses random ones), the curve is
#: monotone out to 120 deg and the 180 deg error halves, -2.88 -> -1.83 A, with
#: near-native calibration unchanged (+0.22 -> +0.25 at 15 deg).
E250_FIT = Variant(
    name="e250_fit",
    description=(
        "As e250_listwise_label, on a clash-aware corpus with "
        "in-place-rotation decoys."
    ),
    generation=GenerationCkpts(vqvae=_E250_VQ, lm=None, refiner=None),
    rescoring=RescoringCkpts(
        vqvae=_E250_VQ,
        mlm=_E250_MLM,
        heads=(HeadSpec("head_fitB", "fit"),),
        codebook_size=8192,
    ),
    affinity=AffinityCkpts(vqvae=_E250_VQ, mlm=_E250_MLM, heads=(), codebook_size=8192),
)


#: :data:`E250_FIT` retrained on a corpus that carries the large ligands the old
#: one threw away, and that no single cofactor is allowed to dominate.
#:
#: ``tokenize_decoys.py`` capped ligands at 50 heavy atoms, so the four CASF
#: targets above that (3ag9 at 67, 3uri at 65, 1u1b at 51, 3prs at 50) sat
#: outside the training distribution entirely -- and all four failed, against
#: one for RTMScore (Fisher p=0.00002). The failure rate tracked the inverse of
#: the corpus density almost exactly: 2.7% where the corpus holds 29% of its
#: mass, 22% at 10%, 29% at 5%, 100% at 0.3%.
#:
#: Raising the cap alone is not enough. Among BioLiP sites with >=40 heavy
#: atoms, HEM alone is 20% and the top ten CCDs are 57%, so the large end of a
#: naive draw is cofactors -- while CASF's large ligands are flexible
#: peptidomimetics. ``--max-per-ccd 20`` caps what any one ligand contributes,
#: which triples the peptide-like share (355 -> 1070 complexes) and, because the
#: duplicates it drops free up the draw, raises the complex count from 20k to
#: 32k at the same time.
#:
#: Measured on 95 CASF targets over 8 frames, against the e250_fit corpus:
#: rho +0.914 -> +0.924, the 30-39 heavy-atom band's failure rate 15% -> 10%,
#: and the <30 band (73% of CASF) held at 1% -- which the cap-only arm did not
#: manage (it traded <30 up to 3% for 30-39 down to 5%).
E250_DIV = Variant(
    name="e250_div",
    description=(
        "As e250_fit, on a corpus with ligands up to 80 heavy atoms and "
        "at most 20 sites per CCD."
    ),
    generation=GenerationCkpts(vqvae=_E250_VQ, lm=None, refiner=None),
    rescoring=RescoringCkpts(
        vqvae=_E250_VQ,
        mlm=_E250_MLM,
        heads=(HeadSpec("head_div", "div"),),
        codebook_size=8192,
    ),
    affinity=AffinityCkpts(vqvae=_E250_VQ, mlm=_E250_MLM, heads=(), codebook_size=8192),
)


#: :data:`E250_DIV` retrained from a different seed, to report the arm's spread
#: rather than one draw of it.
#:
#: The three seeds land 0.017 apart in validation loss (3.7481 / 3.7306 /
#: 3.7366, sd 0.0089) and their predictions agree at Spearman 0.991, yet the
#: pose they pick differs on 8.8% of targets. The seed that was scored first
#: (7) is the worst of the three on validation, so its CASF numbers are a low
#: draw, not the arm's centre. Declared before reading: the three are reported
#: as a mean, and the best of them is not taken.
E250_DIV_S8 = Variant(
    name="e250_div_s8",
    description="e250_div, seed 8.",
    generation=GenerationCkpts(vqvae=_E250_VQ, lm=None, refiner=None),
    rescoring=RescoringCkpts(
        vqvae=_E250_VQ, mlm=_E250_MLM,
        heads=(HeadSpec("head_div_s8", "div_s8"),), codebook_size=8192,
    ),
    affinity=AffinityCkpts(vqvae=_E250_VQ, mlm=_E250_MLM, heads=(), codebook_size=8192),
)

#: :data:`E250_DIV` from a third seed; see :data:`E250_DIV_S8`.
E250_DIV_S9 = Variant(
    name="e250_div_s9",
    description="e250_div, seed 9.",
    generation=GenerationCkpts(vqvae=_E250_VQ, lm=None, refiner=None),
    rescoring=RescoringCkpts(
        vqvae=_E250_VQ, mlm=_E250_MLM,
        heads=(HeadSpec("head_div_s9", "div_s9"),), codebook_size=8192,
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
        heads=(HeadSpec("head_big_lw1.0_s7", "listwise_big"),),
        codebook_size=8192,
    ),
    affinity=AffinityCkpts(
        vqvae=_E250_VQ_BIG, mlm=_E250_MLM, heads=(), codebook_size=8192
    ),
)



#: Same head and same loss as :data:`E250_LISTWISE`; the decoy corpus is what
#: changed. 6 of its 16 decoys per complex now rotate a subset of torsions,
#: bisected to land at 0.3-1.5 A with the rest of the molecule left in place.
#:
#: The 2026-08-20 failure analysis found one structural mismatch between this
#: corpus and CASF, and only one. In the band where docking power is decided
#: (0-1 A) the coefficient of variation of per-atom displacement is 0.64 on
#: CASF and was 0.40 here: CASF's near-native decoys are "most atoms perfect,
#: one group swung out", this corpus's were "whole molecule nudged". 90% of the
#: arm's CASF mistakes are internal-conformation errors (2.41 A of torsion
#: against 0.88 A of centroid) on flexible ligands (7 rotatable bonds against
#: 5, the only ligand property that separates failures from successes), so the
#: head was being asked to make a call it had never been trained to make. The
#: rebuilt corpus measures 0.61 in that band.
E250_LISTWISE_TORSION = Variant(
    name="e250_listwise_torsion",
    description=(
        "As e250_listwise, but the decoy corpus adds near-native "
        "torsion-only poses matching CASF's 0-1 A displacement pattern."
    ),
    generation=GenerationCkpts(vqvae=_E250_VQ, lm=None, refiner=None),
    rescoring=RescoringCkpts(
        vqvae=_E250_VQ,
        mlm=_E250_MLM,
        heads=(HeadSpec("head_ntor_s7", "torsion"),),
        codebook_size=8192,
    ),
    affinity=AffinityCkpts(vqvae=_E250_VQ, mlm=_E250_MLM, heads=(), codebook_size=8192),
)



#: Same head, same loss, same decoy corpus as :data:`E250_LISTWISE`; the MLM
#: backbone underneath it trained longer.
#:
#: The published backbone stopped at ``--max-epochs 3`` with its val loss still
#: falling 9.5% in that last epoch -- a cap, not convergence -- so every head was
#: being fine-tuned on a representation that was still improving fast. Continuing
#: it took three attempts: twice the run died mid-epoch (at 84% and at 92%) and
#: left nothing, because Lightning only checkpoints at validation and validation
#: only ran at epoch end. The third run validates every quarter epoch and drops
#: the peak LR to 1e-4 -- ``--init-from`` restores weights but not the optimizer
#: or the schedule, so the default 4e-4 warmup knocks an annealed model out of
#: its minimum, which is what the second attempt's 0.9235 was. Val went
#: 0.8009 -> 0.7871, still falling at the end.
_E250_MLM_LONG = (
    "pocket-ligand-mlm/mlm_e250lig3_ck/checkpoints/mlm-e00s057122-vl0.7871.ckpt"
)

E250_LISTWISE_MLM2 = Variant(
    name="e250_listwise_mlm2",
    description=(
        "As e250_listwise, on a backbone whose truncated pretraining was "
        "continued (val 0.8009 -> 0.7871)."
    ),
    generation=GenerationCkpts(vqvae=_E250_VQ, lm=None, refiner=None),
    rescoring=RescoringCkpts(
        vqvae=_E250_VQ,
        mlm=_E250_MLM_LONG,
        heads=(HeadSpec("head_mlm2_s7", "mlm2"),),
        codebook_size=8192,
    ),
    affinity=AffinityCkpts(
        vqvae=_E250_VQ, mlm=_E250_MLM_LONG, heads=(), codebook_size=8192
    ),
)



#: Same head, same backbone, same corpus as :data:`E250_LISTWISE`; the listwise
#: softmax is restricted to the 5 poses the head currently ranks best.
#:
#: On the 23 CASF targets it loses and RTMScore wins, this head's top-10
#: overlaps RTMScore's by 6 of 10 -- the same overlap it has where it wins. It
#: is finding the right candidates and mis-ordering the top of them: on 1mq6 its
#: top five are 2.1, 1.0, 0.5, 0.8, 0.7 A, on 3u8n 2.7, 0.7, 2.5, 0.4, 0.9.
#: Counting where the first sub-2 A pose sits, 24 of the 27 failures have one at
#: rank 2-5; only 3 (all ligands of 47-65 heavy atoms) have their top group
#: collapse. A softmax over all ~80 poses spends most of its gradient telling
#: 8 A decoys from each other, so k=5 is where the contest that decides docking
#: power actually is -- and it covers every one of those 24.
E250_LISTWISE_TOPK = Variant(
    name="e250_listwise_topk",
    description=(
        "As e250_listwise, with the listwise softmax restricted to the "
        "head's top 5 poses."
    ),
    generation=GenerationCkpts(vqvae=_E250_VQ, lm=None, refiner=None),
    rescoring=RescoringCkpts(
        vqvae=_E250_VQ,
        mlm=_E250_MLM,
        heads=(HeadSpec("head_topk5", "topk5"),),
        codebook_size=8192,
    ),
    affinity=AffinityCkpts(vqvae=_E250_VQ, mlm=_E250_MLM, heads=(), codebook_size=8192),
)



#: :data:`E250_LISTWISE`'s loss plus a SECOND listwise term over the head's top
#: 5 poses, at equal weight. Same head, same backbone, same corpus.
#:
#: Replacing the full-set term with the top-5 one (``e250_listwise_topk``)
#: measured 90.5 -> 89.5, and the way it failed is the argument for this arm:
#: it won 10 targets, every one of them from the 24 whose answer sat at rank
#: 2-5, and lost 13 that the full-set term had been ordering correctly. The two
#: jobs are separable -- rejecting the far decoys keeps the global ranking,
#: separating the near ones decides docking power -- so the top-5 term is added
#: rather than substituted. Equal weight because nothing in the analysis says
#: which should dominate.
E250_LISTWISE_ADD = Variant(
    name="e250_listwise_add",
    description=(
        "As e250_listwise, plus an equally weighted listwise term over the "
        "head's top 5 poses."
    ),
    generation=GenerationCkpts(vqvae=_E250_VQ, lm=None, refiner=None),
    rescoring=RescoringCkpts(
        vqvae=_E250_VQ,
        mlm=_E250_MLM,
        heads=(HeadSpec("head_add1.0", "add"),),
        codebook_size=8192,
    ),
    affinity=AffinityCkpts(vqvae=_E250_VQ, mlm=_E250_MLM, heads=(), codebook_size=8192),
)



#: :data:`E250_LISTWISE`'s loss plus a listwise term over the 5 poses with the
#: LOWEST RMSD -- chosen by the label, not by the head's own ranking.
#:
#: Two model-side variants came first and both lost: restricting the softmax to
#: the head's top 5 gave 89.5, adding it at equal weight gave 87.7. The more
#: weight the self-selected term carried the worse it got, which rules out a
#: gradient-budget explanation and leaves feedback -- early in training the head
#: picks the wrong five and then sharpens itself on them. Selecting by label
#: fixes the comparison set. It is used only to choose which poses enter the
#: term during training; inference scores every pose as before.
#:
#: The band this targets is the one that decides the benchmark and the one the
#: head is worst at: predictions for poses truly at 0.5-1 A have a standard
#: deviation of 0.67 A, wider than the band itself, and the rank correlation
#: inside 0-1 A is +0.29 against +0.84 over the full range. 150 of ProLIT's 284
#: CASF picks land in that band (RTMScore: 133, with 82 rather than 55 in
#: 0-0.5 A). The top-5 by label span a median of 0.30 A, i.e. entirely inside it.
E250_LISTWISE_LABEL = Variant(
    name="e250_listwise_label",
    description=(
        "As e250_listwise, plus a listwise term over the 5 lowest-RMSD poses."
    ),
    generation=GenerationCkpts(vqvae=_E250_VQ, lm=None, refiner=None),
    rescoring=RescoringCkpts(
        vqvae=_E250_VQ,
        mlm=_E250_MLM,
        heads=(HeadSpec("head_lab_k5_w1.0", "label"),),
        codebook_size=8192,
    ),
    affinity=AffinityCkpts(vqvae=_E250_VQ, mlm=_E250_MLM, heads=(), codebook_size=8192),
)



#: :data:`E250_LISTWISE_LABEL` with the label-selected set narrowed from 5 to 3.
#:
#: k=5 was the first of these terms to help: 90.5 -> 91.2 DP@2A and 72.3 -> 75.1
#: DP@1A, the latter level with RTMScore's 75.4. What remains is a narrower
#: contest than k=5 trains for. Of the 157 targets whose best available pose is
#: under 0.5 A, this head takes a sub-0.5 A pose on 58 and RTMScore on 82; on
#: the 38 where RTMScore wins that band and it does not, it picks a median
#: 0.71 A pose while a 0.34 A one sits available -- every one of the 38. The
#: top-5 by label span 0.30 A; the top-3 span roughly half that, which is the
#: resolution those 38 turn on.
E250_LISTWISE_LABEL3 = Variant(
    name="e250_listwise_label3",
    description=(
        "As e250_listwise_label, with the label-selected set narrowed to 3."
    ),
    generation=GenerationCkpts(vqvae=_E250_VQ, lm=None, refiner=None),
    rescoring=RescoringCkpts(
        vqvae=_E250_VQ,
        mlm=_E250_MLM,
        heads=(HeadSpec("head_lab2_k3_w1.0", "label3"),),
        codebook_size=8192,
    ),
    affinity=AffinityCkpts(vqvae=_E250_VQ, mlm=_E250_MLM, heads=(), codebook_size=8192),
)



#: :data:`E250_LISTWISE_LABEL` with a sharper temperature on the top-k term.
#:
#: The five label-selected poses span a median 0.30 A, and at the shared tau of
#: 0.5 the target softmax over that range is nearly flat -- the term asks for an
#: ordering while barely preferring one. Narrowing the set instead (k=3) raised
#: the count of sub-0.5 A picks 58 -> 62 but cost DP@2A 91.2 -> 89.5, because
#: poses dropped from the set stop being ordered at all. Sharpening keeps all
#: five in play and widens the target gaps between them.
E250_LISTWISE_SHARP = Variant(
    name="e250_listwise_sharp",
    description="As e250_listwise_label, with the top-k term's tau at 0.15.",
    generation=GenerationCkpts(vqvae=_E250_VQ, lm=None, refiner=None),
    rescoring=RescoringCkpts(
        vqvae=_E250_VQ,
        mlm=_E250_MLM,
        heads=(HeadSpec("head_st0.15", "sharp"),),
        codebook_size=8192,
    ),
    affinity=AffinityCkpts(vqvae=_E250_VQ, mlm=_E250_MLM, heads=(), codebook_size=8192),
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
        E250_LISTWISE,
        E250_LISTWISE_BIG,
        E250_LISTWISE_TORSION,
        E250_LISTWISE_MLM2,
        E250_LISTWISE_TOPK,
        E250_LISTWISE_ADD,
        E250_LISTWISE_LABEL,
        E250_LISTWISE_LABEL3,
        E250_LISTWISE_SHARP,
        E250_FIT,
        E250_DIV,
        E250_DIV_S8,
        E250_DIV_S9,
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

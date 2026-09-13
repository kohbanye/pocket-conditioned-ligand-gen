"""Which weights each affinity arm means -- the single definition.

An affinity arm is three checkpoints: the tokenizer that turns a complex into
codes, the MLM that reads them, and the head that maps the pooled ligand
representation to pK. They travel together. A head is trained against one
tokenizer's code space, so pairing it with another silently produces plausible
numbers from a vocabulary the weights were never fitted to.

This registry used to be a field inside ``pose_rescoring_bench.variants``,
alongside the pose and generation checkpoints of the same tokenizer arms. It
moved here whole rather than being copied: the affinity head and the pose head
of one arm share neither a backbone (``wxlhgqx3`` vs ``j90rlrgm`` for ``joint``)
nor a corpus, so a single registry served no shared fact, and two registries
naming the same arm are how two tables end up describing different models.
The tokenizer *arm identity* -- joint vs separate, which VQ runs, which
normalization statistics -- is still defined once in
:mod:`prolit_bench.variants`, and is not restated here.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The tokenizer the paper reports: the joint all-atom VQ retrained to epoch
#: 237 (val atom_coord 0.1021, against 0.1073 for the first-generation
#: ``xzkjxu9q``), with an MLM pretrained on a corpus holding out all 9,329
#: evaluation PDBs rather than only the CASF core.
E250_VQ = (
    "pocket-ligand-vqvae/vq_e250_lig3/checkpoints/"
    "atomvqvae-epoch=237-val/atom_coord=0.1021.ckpt"
)
E250_MLM = "pocket-ligand-mlm/mlm_e250lig3/checkpoints/mlm-e02-vl0.8009.ckpt"

#: The first-generation tokenizer every published affinity number was measured
#: on, kept so the e250 arms have something to be compared against.
V1_VQ = (
    "pocket-ligand-vqvae/xzkjxu9q/checkpoints/"
    "atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
)
#: The leak-free backbone: CASF fully excluded from its pretraining corpus. The
#: affinity numbers use it rather than the higher-scoring ``j90rlrgm``, which
#: saw interface data that includes CASF. The MLM never sees an affinity label,
#: so a pK leak is not possible either way -- this rules out the structural
#: memorization that would be.
V1_MLM_LEAKFREE = "pocket-ligand-mlm/wxlhgqx3/checkpoints/mlm-e02-vl0.8199.ckpt"


@dataclass(frozen=True)
class HeadSpec:
    """One affinity head: a checkpoint, and the label its dump is named after.

    ``ckpt`` is either an exact path ending in ``.ckpt``, relative to the source
    repo, or a training run name resolved to that run's lowest-val-loss
    checkpoint by :func:`prolit_bench.runs.resolve_scoring_head`.
    """

    ckpt: str
    label: str


@dataclass(frozen=True)
class Variant:
    """One affinity arm: a tokenizer, a backbone, and the head(s) on top.

    ``heads`` is a tuple because an arm may be reported as a fixed z-sum
    ensemble. The paper's main table is a single head; more than one entry here
    is an ablation, never the headline number.
    """

    name: str
    description: str
    mlm: str
    vqvae: str | None = None
    heads: tuple[HeadSpec, ...] = ()
    codebook_size: int = 8192
    #: Separate-tokenizer ablation arm: protein-only + ligand-only VQ-VAEs
    #: stitched into one code space of ``codebook_size`` total.
    protein_vqvae: str | None = None
    ligand_vqvae: str | None = None
    protein_norm: str | None = None
    ligand_norm: str | None = None
    notes: str = ""

    @property
    def is_separate(self) -> bool:
        return self.protein_vqvae is not None

    def require_heads(self) -> tuple[HeadSpec, ...]:
        """The arm's heads, or a message saying which arm has none trained yet."""
        if not self.heads:
            msg = (
                f"affinity arm {self.name!r} has no head checkpoints yet. "
                "Train one against its tokenizer before evaluating it."
            )
            raise ValueError(msg)
        return self.heads


#: The published arm: every affinity number in ``docs/results/`` was measured
#: here. Its head is mean-pooled and trained on BioLiP Kd/Ki labels only.
V1_KDKI_MEAN = Variant(
    name="v1_kdki_mean",
    description="xzkjxu9q + leak-free MLM + one mean-pooled Kd/Ki head.",
    vqvae=V1_VQ,
    mlm=V1_MLM_LEAKFREE,
    heads=(
        HeadSpec(
            "pocket-ligand-rescore/tzqaubl4/checkpoints/rescore-e09-vl0.6196.ckpt",
            "kdki-mean",
        ),
    ),
    notes="the single-model row: scoring R 0.771, ranking rho 0.647",
)

#: The two arms of the rotation-augmentation test (2026-09-13). Both are the
#: published recipe -- one mean-pooled head, pK regression, Kd/Ki labels only --
#: on the current tokenizer, and differ ONLY in how many orientations of each
#: training complex the corpus holds. Heads are named by training run rather
#: than by path because the run does not exist yet when this is written;
#: :func:`prolit_bench.runs.resolve_scoring_head` picks its lowest-val-loss
#: checkpoint.
E250_ROT1 = Variant(
    name="e250_rot1",
    description="vq_e250_lig3 + its MLM + one mean-pooled Kd/Ki head, 1 frame/complex.",
    vqvae=E250_VQ,
    mlm=E250_MLM,
    heads=(HeadSpec("aff_e250_r1", "kdki-mean"),),
    notes="control for the rotation-augmentation test",
)

E250_ROT8 = Variant(
    name="e250_rot8",
    description="Same, on a corpus holding 8 random orientations of each complex.",
    vqvae=E250_VQ,
    mlm=E250_MLM,
    heads=(HeadSpec("aff_e250_r8", "kdki-mean"),),
    notes="rotation augmentation: 8x the documents, same complexes",
)

#: The listwise arm. Same corpus and head as :data:`E250_ROT1`; the only change
#: is a ListNet term over each protein's ligands, which is the definition of
#: ranking power rather than a proxy for it.
#:
#: A ranking loss was tried on this corpus in July and lost on both metrics --
#: the encoder memorized the training order of 8.5k documents. Two things are
#: different now. The corpus is the same size, so that risk is unchanged and
#: ``train/list`` collapsing to zero is still the signal that it happened. But
#: the term's sign was wrong for a pK label: ListNet matches a softmax over the
#: labels, and without ``--listwise-higher-is-better`` that softmax puts its
#: mass on the WEAKEST binder in each group. The weight is half of what failed
#: then, and it is one point, not a sweep -- picking a weight by CASF score
#: would be selecting on the test set.
E250_LISTWISE = Variant(
    name="e250_listwise",
    description="Same as e250_rot1, plus a sign-corrected ListNet term per protein.",
    vqvae=E250_VQ,
    mlm=E250_MLM,
    heads=(HeadSpec("aff_e250_lw", "kdki-mean"),),
    notes="listwise weight 0.5, higher-is-better; groups are UniProt proteins",
)

#: The frozen-encoder arm. Same corpus and loss as :data:`E250_ROT8`; only the
#: 0.59M head is trained, against 99.6M when the encoder moves.
#:
#: Added after watching ``afftrain_rot-8`` reach train/loss 0.034 against
#: val/loss 0.922 with its BEST checkpoint at epoch 0 -- the encoder memorizes
#: 65k documents inside one epoch. Rotations multiplied the documents without
#: adding a single independent label, so they were never going to stop that;
#: what they can do is teach frame robustness, which is a different claim.
#:
#: Freezing was tried in July and lost, but it was switched on together with a
#: ranking loss, so which of the two cost the points was never separated. Here
#: the objective stays plain regression.
E250_FROZEN = Variant(
    name="e250_frozen",
    description="Same as e250_rot8 with the MLM encoder frozen (head-only training).",
    vqvae=E250_VQ,
    mlm=E250_MLM,
    heads=(HeadSpec("aff_e250_frz", "kdki-mean"),),
    notes="0.59M trainable; tests whether memorization is a cost or a necessity",
)

#: Two more training seeds of the best arm. The night's ranking of arms rested
#: on one training seed each, and the inference-side seed spread (+/-0.005) says
#: nothing about the training-side one, which is usually the larger of the two.
E250_ROT1_S11 = Variant(
    name="e250_rot1_s11",
    description="e250_rot1 retrained with seed 11.",
    vqvae=E250_VQ,
    mlm=E250_MLM,
    heads=(HeadSpec("aff_e250_r1_s11", "kdki-mean"),),
)
E250_ROT1_S12 = Variant(
    name="e250_rot1_s12",
    description="e250_rot1 retrained with seed 12.",
    vqvae=E250_VQ,
    mlm=E250_MLM,
    heads=(HeadSpec("aff_e250_r1_s12", "kdki-mean"),),
)

#: BioLIP Kd/Ki UNION PDBbind v2020 Kd/Ki: 15,743 training documents against
#: 8,150. The two sources overlap on only 4,218 PDB ids, so PDBbind contributes
#: 4,570 complexes BioLIP's parse never produced -- a 54% increase in
#: independent labels, which is the one thing rotations did not add.
#:
#: July compared PDBbind against BioLIP as a REPLACEMENT and found them
#: equivalent, and concluded data quality was not the bottleneck. That says
#: nothing about their union, which is what this is.
#:
#: The rerun of the union with BioLIP's val pdbs excluded from PDBbind too, so
#: the two arms are selected on a validation set neither has trained on.
E250_UNION_CLEAN = Variant(
    name="e250_union_clean",
    description="e250 + BioLIP union PDBbind Kd/Ki, val pdbs excluded from both.",
    vqvae=E250_VQ,
    mlm=E250_MLM,
    heads=(HeadSpec("aff_e250_unionc_s7", "kdki-mean"),),
    notes="the arm that carries the H7 verdict",
)
E250_UNION_CLEAN_S11 = Variant(
    name="e250_union_clean_s11",
    description="e250_union_clean retrained with seed 11.",
    vqvae=E250_VQ,
    mlm=E250_MLM,
    heads=(HeadSpec("aff_e250_unionc_s11", "kdki-mean"),),
)
E250_UNION_CLEAN_S12 = Variant(
    name="e250_union_clean_s12",
    description="e250_union_clean retrained with seed 12.",
    vqvae=E250_VQ,
    mlm=E250_MLM,
    heads=(HeadSpec("aff_e250_unionc_s12", "kdki-mean"),),
)

#: PDBbind alone, to separate "more labels" from "a different label source".
#:
#: Needed before the union's result can be explained rather than storied: if the
#: union raises scoring R and lowers ranking rho, that is either two sources
#: disagreeing about the same protein's ligands (PDBbind pK median 6.18 against
#: BioLIP's 6.54, which would blur within-cluster order) or simply more data
#: helping a global correlation. This arm tells them apart.
#:
#: Validation is BioLIP's val set, as for every other arm, so all three are
#: selected on the same 431 documents -- and the clean PDBbind corpus excludes
#: those pdbs from training, so it is a held-out set for this arm too.
E250_PDBBIND = Variant(
    name="e250_pdbbind",
    description="e250 + one mean-pooled head on PDBbind v2020 Kd/Ki labels alone.",
    vqvae=E250_VQ,
    mlm=E250_MLM,
    heads=(HeadSpec("aff_e250_pdbb_s7", "kdki-mean"),),
    notes="separates label count from label source in the union result",
)
E250_PDBBIND_S11 = Variant(
    name="e250_pdbbind_s11",
    description="e250_pdbbind retrained with seed 11.",
    vqvae=E250_VQ,
    mlm=E250_MLM,
    heads=(HeadSpec("aff_e250_pdbb_s11", "kdki-mean"),),
)
E250_PDBBIND_S12 = Variant(
    name="e250_pdbbind_s12",
    description="e250_pdbbind retrained with seed 12.",
    vqvae=E250_VQ,
    mlm=E250_MLM,
    heads=(HeadSpec("aff_e250_pdbb_s12", "kdki-mean"),),
)

#: H9: the union corpus plus a sign-corrected ListNet term over protein groups.
#:
#: Listwise has failed twice (July, and H4 today), but both attempts ran on the
#: rot1-sized corpus, which has R sd 0.05. The union corpus has twice the data
#: and 17x lower scoring variance, and ranking is the metric furthest from
#: GenScore (0.663 against 0.735), so a ranking-targeted loss is the matched
#: instrument. Known dilution: PDBbind contributes no ``.grp``, so ``mix.py``
#: gave it singleton groups -- only the BioLiP half supplies ranking pairs.
E250_UNION_LW = Variant(
    name="e250_union_lw",
    description="e250_union_clean plus a sign-corrected ListNet term per protein.",
    vqvae=E250_VQ,
    mlm=E250_MLM,
    heads=(HeadSpec("aff_e250_unionlw_s7", "kdki-mean"),),
    notes="listwise weight 0.5, higher-is-better; reference is e250_union_clean",
)
E250_UNION_LW_S11 = Variant(
    name="e250_union_lw_s11",
    description="e250_union_lw retrained with seed 11.",
    vqvae=E250_VQ, mlm=E250_MLM,
    heads=(HeadSpec("aff_e250_unionlw_s11", "kdki-mean"),),
)
E250_UNION_LW_S12 = Variant(
    name="e250_union_lw_s12",
    description="e250_union_lw retrained with seed 12.",
    vqvae=E250_VQ, mlm=E250_MLM,
    heads=(HeadSpec("aff_e250_unionlw_s12", "kdki-mean"),),
)

#: H8: the union with the 151 contradictory-label complexes removed.
E250_UNION_NOCONF = Variant(
    name="e250_union_noconf",
    description="e250_union_clean minus complexes whose two sources disagree on pK.",
    vqvae=E250_VQ, mlm=E250_MLM,
    heads=(HeadSpec("aff_e250_unionnc_s7", "kdki-mean"),),
    notes="tests whether contradictory labels are what cost union its ranking",
)
E250_UNION_NOCONF_S11 = Variant(
    name="e250_union_noconf_s11",
    description="e250_union_noconf retrained with seed 11.",
    vqvae=E250_VQ, mlm=E250_MLM,
    heads=(HeadSpec("aff_e250_unionnc_s11", "kdki-mean"),),
)
E250_UNION_NOCONF_S12 = Variant(
    name="e250_union_noconf_s12",
    description="e250_union_noconf retrained with seed 12.",
    vqvae=E250_VQ, mlm=E250_MLM,
    heads=(HeadSpec("aff_e250_unionnc_s12", "kdki-mean"),),
)

REGISTRY: dict[str, Variant] = {
    v.name: v
    for v in (
        E250_UNION_LW,
        E250_UNION_LW_S11,
        E250_UNION_LW_S12,
        E250_UNION_NOCONF,
        E250_UNION_NOCONF_S11,
        E250_UNION_NOCONF_S12,
        E250_PDBBIND,
        E250_PDBBIND_S11,
        E250_PDBBIND_S12,
        E250_UNION_CLEAN,
        E250_UNION_CLEAN_S11,
        E250_UNION_CLEAN_S12,
        E250_ROT1_S11,
        E250_ROT1_S12,
        V1_KDKI_MEAN,
        E250_ROT1,
        E250_ROT8,
        E250_LISTWISE,
        E250_FROZEN,
    )
}

#: Which registered arms are the same experiment at a different training seed.
#:
#: Stated as data rather than recovered from directory names, because the two
#: kinds of seed are spelled alike and mean opposite things: ``_s11`` here is a
#: TRAINING seed (a different head), while ``_f16s1`` on a dump directory is an
#: INFERENCE seed (the same head, different rotations). Parsing would conflate
#: them, and conflating them is how a single run gets read as a measurement.
#:
#: Retraining with nothing changed but this moves scoring R by ~0.05 and
#: ranking rho by ~0.04, so an arm compared at one seed says nothing. Families
#: are compared to each other seed by seed; see
#: :func:`affinity_bench.aggregate.paired_by_seed`.
#: ``e250_union`` is deliberately absent: its val set was 60% contaminated, so a
#: paired row against it would report a contaminated comparison as the verdict.
#: Its dumps stay in the per-arm table, where they are labelled, and the clean
#: rerun takes its place here.
SEED_FAMILIES: dict[str, dict[int, str]] = {
    "e250_rot1": {7: "e250_rot1", 11: "e250_rot1_s11", 12: "e250_rot1_s12"},
    "e250_union_clean": {
        7: "e250_union_clean",
        11: "e250_union_clean_s11",
        12: "e250_union_clean_s12",
    },
    "e250_pdbbind": {
        7: "e250_pdbbind",
        11: "e250_pdbbind_s11",
        12: "e250_pdbbind_s12",
    },
    "e250_union_lw": {
        7: "e250_union_lw",
        11: "e250_union_lw_s11",
        12: "e250_union_lw_s12",
    },
    "e250_union_noconf": {
        7: "e250_union_noconf",
        11: "e250_union_noconf_s11",
        12: "e250_union_noconf_s12",
    },
}

#: The family a paired comparison is measured against by default: the current
#: best recipe with no extra data and no extra loss.
SEED_REFERENCE = "e250_rot1"

#: Families whose correct control is NOT the default.
#:
#: A paired comparison is only interpretable against an arm that differs by one
#: thing. ``e250_union_lw`` changes the loss *on top of* the union corpus, and
#: ``e250_union_noconf`` changes which complexes that corpus contains -- so both
#: must be read against ``e250_union_clean``. Compared against ``e250_rot1``
#: instead, each would silently inherit the union corpus's +0.035 scoring gain
#: and a loss change would be credited for a data change.
SEED_REFERENCE_FOR: dict[str, str] = {
    "e250_union_lw": "e250_union_clean",
    "e250_union_noconf": "e250_union_clean",
}


#: Order the comparison table lists our arms in.
ARM_ORDER: tuple[str, ...] = (
    "e250_union_noconf",
    "e250_union_lw",
    "e250_union_clean",
    "e250_pdbbind",
    "e250_frozen",
    "e250_listwise",
    "e250_rot8",
    "e250_rot1",
    "v1_kdki_mean",
)


def get(name: str) -> Variant:
    """Look up an affinity arm by name."""
    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        msg = f"unknown affinity arm {name!r}; known: {known}"
        raise KeyError(msg)
    return REGISTRY[name]


def register(variant: Variant) -> Variant:
    """Add an arm to the registry.

    The improvement loop trains heads faster than a paper table changes, so new
    arms arrive here as data. Only arms the paper reports should survive in the
    module; the rest belong in their run directory's ``run.json``.
    """
    if variant.name in REGISTRY:
        msg = f"affinity arm {variant.name!r} is already registered"
        raise KeyError(msg)
    REGISTRY[variant.name] = variant
    return variant


__all__ = [
    "ARM_ORDER",
    "E250_MLM",
    "E250_VQ",
    "REGISTRY",
    "SEED_FAMILIES",
    "SEED_REFERENCE",
    "SEED_REFERENCE_FOR",
    "HeadSpec",
    "Variant",
    "get",
    "register",
]

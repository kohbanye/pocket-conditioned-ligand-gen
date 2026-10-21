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
class StapledSpec:
    """The ESM3 x ConfSeq baseline's stand-ins for a VQ-VAE.

    Paths relative to the source repo, as in ``pose_rescoring_bench``'s spec of
    the same name: the ESM3 structure-token cache keyed by CASF target id, the
    ConfSeq checkout, and the frozen ConfSeq vocabulary every stapled corpus
    was written with.
    """

    esm3_cache: str = "data/esm3_tokens_casf2016"
    confseq_repo: str = "third_party/ConfSeq"
    confseq_vocab: str = "data/stapled/confseq_vocab.json"


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
    #: The inputs this arm's head was TRAINED on, so evaluation cannot silently
    #: use different ones. ``pocket_distance`` selects pocket residues by CA
    #: distance ("ca") or by minimum heavy-atom distance ("heavy"); the latter
    #: keeps side chains that touch the ligand but whose CA is far. ``chemistry``
    #: reads formal charge and bond orders from a validated source ("validated")
    #: instead of rebuilding the molecule from elements ("legacy"), which cannot
    #: recover charge at all. A head trained under one and scored under the
    #: other is a different model, and no column in a dump would reveal it.
    pocket_distance: str = "ca"
    chemistry: str = "legacy"
    #: Which encoder states this arm's head was trained to read. Parameter-free,
    #: so a checkpoint loads under the wrong one without any shape error and
    #: quietly scores a model that was never trained.
    readout: str = "ligand"
    #: Whether the head reads a learned mix over every encoder layer instead of
    #: the last layer only. Like ``readout`` it is a property of how the
    #: checkpoint was TRAINED: the mix adds weights to the state_dict, so a
    #: mismatch fails loudly instead of scoring through a path never trained.
    layer_mix: bool = False
    #: LoRA rank the head was TRAINED with (0 = none). Like readout and
    #: layer_mix it changes the state_dict, so a mismatch fails loudly.
    lora_rank: int = 0
    #: Set for the ESM3 x ConfSeq baseline, which has no VQ-VAE: pocket codes
    #: come from an ESM3 cache and the ligand from ConfSeq plus 4 pose tokens.
    stapled: StapledSpec | None = None
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

#: The 2026-09-14 input-repair round: one head per (input condition, seed), all
#: on E250 and all trained on the same 7,714-site development cohort.
#:
#: Two input defects were found by reading the structures rather than the
#: residuals, and they are independent of each other. The ligand's formal charge
#: never reached the descriptor, because the molecule was rebuilt from elements
#: and bond orders -- the 285 CASF core ligands carry 212 charged atoms and the
#: rebuilt path keeps 14. And pocket residues were selected by CA distance,
#: which drops side chains that touch the ligand: 124 of 283 complexes lose a
#: residue lying within 4 A of it, and raising the 50-residue cap fixes none of
#: them. On the development set, repairing both moved Pearson +0.030 against
#: this round's own control, against +0.008 and +0.003 for each alone.
_DEV_ROUND = "experiments/affinity_loop_20260914/runs_fresh"

#: arm -> (run directory, pocket distance, chemistry, {training seed: ckpt}).
#: Stated as one table because sixteen arms differing in two fields and a seed
#: are data, and sixteen near-identical blocks are somewhere to hide a typo.
_DEV_ROUND_ARMS: dict[str, tuple[str, str, str, dict[int, str]]] = {
    "ca_legacy": (
        "original",
        "ca",
        "legacy",
        {
            7: "best-epoch=13",
            11: "best-epoch=11",
            12: "best-epoch=07",
            13: "best-epoch=04",
        },
    ),
    "ca_validated": (
        "ca_validated",
        "ca",
        "validated",
        {
            7: "best-epoch=06",
            11: "best-epoch=11",
            12: "best-epoch=09",
            13: "best-epoch=04",
        },
    ),
    "heavy_legacy": (
        "heavy_legacy",
        "heavy",
        "legacy",
        {
            7: "best-epoch=19",
            11: "best-epoch=17",
            12: "best-epoch=01",
            13: "best-epoch=04",
        },
    ),
    "heavy_validated": (
        "heavy_validated",
        "heavy",
        "validated",
        {
            7: "best-epoch=06",
            11: "best-epoch=06",
            12: "best-epoch=03",
            13: "best-epoch=04",
        },
    ),
}

#: Why a CASF number from these arms is a diagnostic and not a headline row.
_DEV_NOTES = (
    "development cohort: 20 of its training sites share an exact protein "
    "sequence AND ligand with a CASF core complex, so this arm's CASF score is "
    "leak-flagged. The cleaned 7,666-site cohort exists for the replacement."
)


#: The training seed whose arm keeps the bare family name, so the dev arms are
#: spelled like the older ones (``e250_rot1`` and ``e250_rot1_s11``).
_BARE_NAME_SEED = 7


def _dev_arm_name(arm: str, seed: int) -> str:
    """The arm name for one input condition and training seed."""
    if seed == _BARE_NAME_SEED:
        return f"e250_dev_{arm}"
    return f"e250_dev_{arm}_s{seed}"


DEV_INPUT_ARMS: dict[str, Variant] = {
    _dev_arm_name(arm, seed): Variant(
        name=_dev_arm_name(arm, seed),
        description=(
            f"e250 + one mean-pooled head on the {pocket} pocket / {chem} "
            f"chemistry inputs, development cohort, training seed {seed}."
        ),
        vqvae=E250_VQ,
        mlm=E250_MLM,
        heads=(
            HeadSpec(
                f"{_DEV_ROUND}/{directory}/head_s{seed}/checkpoints/{ckpt}.ckpt",
                "kdki-mean",
            ),
        ),
        pocket_distance=pocket,
        chemistry=chem,
        notes=_DEV_NOTES,
    )
    for arm, (directory, pocket, chem, seeds) in _DEV_ROUND_ARMS.items()
    for seed, ckpt in seeds.items()
}

#: The dev round's seed families, built from the same table.
DEV_SEED_FAMILIES: dict[str, dict[int, str]] = {
    f"e250_dev_{arm}": {seed: _dev_arm_name(arm, seed) for seed in seeds}
    for arm, (_directory, _pocket, _chem, seeds) in _DEV_ROUND_ARMS.items()
}

#: M14: the same heads again with only the READOUT changed -- no parameter is
#: added, the corpus and the VQ are untouched, so nothing about reconstruction
#: or any other task moves. It asks whether the protein states the LM already
#: computes are reachable by the current pooling, which reads ligand rows only.
#:
#: Development effect over the `ca_legacy` control, four paired seeds:
#: balanced +0.0167 R, allatoms +0.0171 R, both 3/4 seeds. Untested on CASF,
#: and its prior is now lower, because the larger dev effect on this same
#: cohort (the input repair, +0.030) did not survive the trip.
_DEV_READOUT_ARMS: dict[str, tuple[str, dict[int, str]]] = {
    "balanced": (
        "balanced",
        {
            7: "best-epoch=11",
            11: "best-epoch=15",
            12: "best-epoch=02",
            13: "best-epoch=09",
        },
    ),
    "allatoms": (
        "allatoms",
        {
            7: "best-epoch=11",
            11: "best-epoch=14",
            12: "best-epoch=12",
            13: "best-epoch=16",
        },
    ),
}

DEV_READOUT_ARM_VARIANTS: dict[str, Variant] = {
    _dev_arm_name(arm, seed): Variant(
        name=_dev_arm_name(arm, seed),
        description=(
            f"e250 + the same 592,897-parameter head read with {arm} pooling, "
            f"development cohort, training seed {seed}."
        ),
        vqvae=E250_VQ,
        mlm=E250_MLM,
        heads=(
            HeadSpec(
                f"experiments/affinity_loop_20260914/runs_m14/{directory}"
                f"/head_s{seed}/checkpoints/{ckpt}.ckpt",
                "kdki-mean",
            ),
        ),
        pocket_distance="ca",
        chemistry="legacy",
        readout=arm,
        notes=_DEV_NOTES,
    )
    for arm, (directory, seeds) in _DEV_READOUT_ARMS.items()
    for seed, ckpt in seeds.items()
}

DEV_READOUT_SEED_FAMILIES: dict[str, dict[int, str]] = {
    f"e250_dev_{arm}": {seed: _dev_arm_name(arm, seed) for seed in seeds}
    for arm, (_directory, seeds) in _DEV_READOUT_ARMS.items()
}

#: The combination the readout round left untested: `union_noconf`, the corpus
#: that leads on scoring R (family mean 0.7971), carrying the readouts that
#: moved ranking rho on the development cohort (allatoms +0.0171 on 4/4 seeds).
#:
#: The readout arms were trained on `ca_legacy`, which is the WEAKER corpus --
#: its own control sits below both `union_noconf` on R and `e250_rot1` on rho --
#: so the readout has never been measured on top of the best corpus. Identical
#: to the `aff_e250_unionnc_s*` recipe except for `--readout`.
#:
#: The two effects are not assumed additive. H7 and H9 both found corpus effects
#: that failed to stack, and the readout's own R gain was 3/4 seeds, not 4/4.
#: The backbone swap. Every affinity number in this project was measured on
#: ``mlm_e250lig3`` epoch 2 (val 0.8009), which stopped because ``--max-epochs 3``
#: ran out, not because it converged: 1.1034 -> 0.8847 -> 0.8009, still falling.
#:
#: ``mlm_e250lig3_lo`` continues that exact checkpoint for one epoch at lr 1e-4
#: and reaches 0.7867 on the SAME token dir and the same 536-batch val set, so
#: the two losses are directly comparable (the base run sharded 28,561 steps
#: across 4 devices; the continuation ran 114,246 on one -- 28,561 x 4). It has
#: existed since 2026-08-20 and no affinity head has ever used it.
#:
#: The lr is load-bearing, not incidental: ``mlm_e250lig3_x2`` is the same
#: continuation at the default lr and went BACKWARDS, to 0.9235.
#:
#: These arms are the ``aff_e250_unionnc_s*`` recipe with ``--mlm-ckpt`` changed
#: and nothing else, so the paired comparison against ``e250_union_noconf``
#: isolates the backbone.
E250_MLM_LO = (
    "pocket-ligand-mlm/mlm_e250lig3_lo/checkpoints/mlm-e00-vl0.7867.ckpt"
)
_BACKBONE_SWAP_SEEDS = (7, 11, 12, 13)


def _unionnc_lo_arm_name(seed: int) -> str:
    """Seed 7 keeps the bare family name, as in every other family here."""
    if seed == _BARE_NAME_SEED:
        return "e250_unionnc_lo"
    return f"e250_unionnc_lo_s{seed}"


BACKBONE_SWAP_VARIANTS: dict[str, Variant] = {
    _unionnc_lo_arm_name(seed): Variant(
        name=_unionnc_lo_arm_name(seed),
        description=(
            "e250 + the union_noconf corpus on the continued MLM backbone "
            f"(val 0.7867 rather than 0.8009), training seed {seed}."
        ),
        vqvae=E250_VQ,
        mlm=E250_MLM_LO,
        heads=(HeadSpec(f"aff_e250_unionnc_lo_s{seed}", "kdki-mean"),),
        notes=(
            "asks whether the affinity deficit is backbone undertraining; "
            "identical to aff_e250_unionnc_s* except for --mlm-ckpt"
        ),
    )
    for seed in _BACKBONE_SWAP_SEEDS
}

BACKBONE_SWAP_SEED_FAMILIES: dict[str, dict[int, str]] = {
    "e250_unionnc_lo": {
        seed: _unionnc_lo_arm_name(seed) for seed in _BACKBONE_SWAP_SEEDS
    }
}

#: The label-type round (2026-09-14). `--affinity-types` defaults to
#: `KD,KI,IC50`; the whole e250 generation passed `KD,KI`, which the builder's
#: own help calls "halves the data". The V1-era corpora show the size of it:
#: `affinity_pdbbind` holds 14,047 complexes against `e250aff_pdbbind_noconf`'s
#: 7,587, and `affinity` holds 15,078 against `e250aff_r1`'s 8,581.
#:
#: PDBbind cannot be rebuilt -- `data/pdbbind` is gone and the builder needs the
#: structure tree, not just labels -- so this pair is BioLiP only. The two
#: corpora were built the same day from the same snapshot, same tokenizer, same
#: exclusions, differing in exactly one flag, rather than reusing `e250aff_r1`
#: as a control (its build script no longer exists, so "identical except" could
#: not be verified).
#:
#: Two reasons to expect something, addressing the two halves of the deficit:
#: more data should help R (which is flat in family count: 0.8086 at 4 families,
#: 0.8112 at 12), and a new family should help rho (which is still climbing at
#: +0.002/family with no saturation).
_AFFINITY_LABEL_ARMS: dict[str, tuple[str, str]] = {
    # arm family -> (head run-name stem, dump label)
    "e250_aff_ic50": ("aff_e250_ic50", "pk-mean"),
    "e250_aff_kdki_ctl": ("aff_e250_kdkictl", "kdki-mean"),
}
_AFFINITY_LABEL_SEEDS = (7, 11, 12, 13)


def _affinity_label_arm_name(family: str, seed: int) -> str:
    """Seed 7 keeps the bare family name, as in every other family here."""
    if seed == _BARE_NAME_SEED:
        return family
    return f"{family}_s{seed}"


AFFINITY_LABEL_VARIANTS: dict[str, Variant] = {
    _affinity_label_arm_name(fam, seed): Variant(
        name=_affinity_label_arm_name(fam, seed),
        description=(
            f"e250 tokenizer + MLM, BioLiP-only affinity corpus "
            f"({'KD/KI/IC50' if 'ic50' in fam else 'KD/KI'}), training seed {seed}."
        ),
        vqvae=E250_VQ,
        mlm=E250_MLM,
        heads=(HeadSpec(f"{stem}_s{seed}", label),),
        notes=(
            "label-type pair: the two corpora differ only in --affinity-types"
        ),
    )
    for fam, (stem, label) in _AFFINITY_LABEL_ARMS.items()
    for seed in _AFFINITY_LABEL_SEEDS
}

AFFINITY_LABEL_SEED_FAMILIES: dict[str, dict[int, str]] = {
    fam: {seed: _affinity_label_arm_name(fam, seed) for seed in _AFFINITY_LABEL_SEEDS}
    for fam in _AFFINITY_LABEL_ARMS
}

#: The union-IC50 round (2026-09-14 ~22:20), pre-registered in
#: docs/notes/2026-09-14_prereg_union_ic50.md before either corpus was built.
#:
#: The source union is the only lever that ever moved R by ~0.02-0.035, and
#: BioLiP IC50 alone traded rho for +0.005 R. Whether PDBbind-curated IC50 does
#: the same inside the union is untested. Each arm is a BioLiP half plus a new
#: PDBbind half; the two arms differ only in --affinity-types, on both halves.
_UNION_LABEL_ARMS: dict[str, tuple[str, str]] = {
    # arm family -> (head run-name stem, dump label)
    "e250_unionic50": ("aff_e250_unionic50", "pk-mean"),
    "e250_unionkdki": ("aff_e250_unionkdki", "kdki-mean"),
}

UNION_LABEL_VARIANTS: dict[str, Variant] = {
    _affinity_label_arm_name(fam, seed): Variant(
        name=_affinity_label_arm_name(fam, seed),
        description=(
            "e250 tokenizer + MLM, BioLiP + PDBbind union "
            f"({'KD/KI/IC50' if 'ic50' in fam else 'KD/KI'} on both halves), "
            f"training seed {seed}."
        ),
        vqvae=E250_VQ,
        mlm=E250_MLM,
        heads=(HeadSpec(f"{stem}_s{seed}", label),),
        notes="union label-type pair: the corpora differ only in --affinity-types",
    )
    for fam, (stem, label) in _UNION_LABEL_ARMS.items()
    for seed in _AFFINITY_LABEL_SEEDS
}

UNION_LABEL_SEED_FAMILIES: dict[str, dict[int, str]] = {
    fam: {seed: _affinity_label_arm_name(fam, seed) for seed in _AFFINITY_LABEL_SEEDS}
    for fam in _UNION_LABEL_ARMS
}

#: The rotation-ensemble round (2026-09-15), pre-registered in
#: docs/notes/2026-09-15_prereg_rotation_ensemble.md before the corpus was built.
#:
#: Rotation sampling on the BioLiP-only corpus gained at family level (dR +0.014,
#: drho +0.019) but not per seed, and the literature pass found that is the
#: normal shape: AK-Score gets its 0.827 from a 30-network ensemble trained with
#: 24-fold rotation augmentation. This arm puts rotations on the union corpus
#: (union_noconf + 7 extra orientations per train complex, k=0 verified
#: byte-identical) and judges the 8-seed family, not single seeds.
_UNION_R8_SEEDS = (7, 11, 12, 13, 14, 15, 16, 17)

UNION_R8_VARIANTS: dict[str, Variant] = {
    _affinity_label_arm_name("e250_unionr8_ens", seed): Variant(
        name=_affinity_label_arm_name("e250_unionr8_ens", seed),
        description=(
            "e250 tokenizer + MLM, union corpus with 8 orientations per complex, "
            f"one drawn per access, training seed {seed}."
        ),
        vqvae=E250_VQ,
        mlm=E250_MLM,
        heads=(HeadSpec(f"aff_e250_unionr8_s{seed}", "kdki-mean"),),
        notes="rotation-ensemble arm: union_noconf plus --rotations-per-complex 8",
    )
    for seed in _UNION_R8_SEEDS
}

UNION_R8_SEED_FAMILIES: dict[str, dict[int, str]] = {
    "e250_unionr8_ens": {
        seed: _affinity_label_arm_name("e250_unionr8_ens", seed)
        for seed in _UNION_R8_SEEDS
    }
}

#: The cross-recipe mixing round (2026-09-15), pre-registered in
#: docs/notes/2026-09-15_prereg_cross_recipe_mix.md before these seeds were
#: trained. Fresh seeds of the two union recipes, kept in their own families so
#: the 8-seed families the earlier verdicts judged stay exactly as judged.
_CROSSMIX_SEEDS = (18, 19, 20, 21, 22, 23, 24, 25)

#: (family, head stem, corpus description) for the two recipes being mixed.
_CROSSMIX_ARMS = (
    ("e250_unionnc_corr_v2", "aff_e250_unionnccorr",
     "union corpus, batch-correlation loss (0.7)"),
    ("e250_unionr8_ens_v2", "aff_e250_unionr8",
     "union corpus with 8 orientations per complex"),
)

CROSSMIX_VARIANTS: dict[str, Variant] = {
    _affinity_label_arm_name(family, seed): Variant(
        name=_affinity_label_arm_name(family, seed),
        description=f"e250 tokenizer + MLM, {corpus}, training seed {seed}.",
        vqvae=E250_VQ,
        mlm=E250_MLM,
        heads=(HeadSpec(f"{stem}_s{seed}", "kdki-mean"),),
        notes="cross-recipe mixing round: fresh seeds, judged as a two-recipe ensemble",
    )
    for family, stem, corpus in _CROSSMIX_ARMS
    for seed in _CROSSMIX_SEEDS
}

CROSSMIX_SEED_FAMILIES: dict[str, dict[int, str]] = {
    family: {seed: _affinity_label_arm_name(family, seed) for seed in _CROSSMIX_SEEDS}
    for family, _stem, _corpus in _CROSSMIX_ARMS
}

#: The third recipe of the mixing round (2026-09-15), pre-registered in
#: docs/notes/2026-09-15_prereg_heavy_third_recipe.md before its corpora
#: finished building. Same head recipe as the ca arms; the pocket is selected by
#: minimum heavy-atom distance instead of CA distance, which decorrelates our
#: errors more than any other input change measured (dev families 0.82-0.88
#: against 0.945 for the rotation recipe).
_UNION_HEAVY_SEEDS = (26, 27, 28, 29, 30, 31, 32, 33)

UNION_HEAVY_VARIANTS: dict[str, Variant] = {
    _affinity_label_arm_name("e250_unionheavy", seed): Variant(
        name=_affinity_label_arm_name("e250_unionheavy", seed),
        description=(
            "e250 tokenizer + MLM, union corpus with heavy-atom pocket selection, "
            f"training seed {seed}."
        ),
        vqvae=E250_VQ,
        mlm=E250_MLM,
        heads=(HeadSpec(f"aff_e250_unionheavy_s{seed}", "kdki-mean"),),
        pocket_distance="heavy",
        notes="third recipe of the mixing round: heavy-atom pocket selection",
    )
    for seed in _UNION_HEAVY_SEEDS
}

UNION_HEAVY_SEED_FAMILIES: dict[str, dict[int, str]] = {
    "e250_unionheavy": {
        seed: _affinity_label_arm_name("e250_unionheavy", seed)
        for seed in _UNION_HEAVY_SEEDS
    }
}

#: The fourth recipe of the mixing round (2026-09-15), pre-registered in
#: docs/notes/2026-09-15_prereg_allatoms_fourth_recipe.md before any head
#: existed. Same corpus and head size as the plain union recipe; the readout
#: pools every atom token instead of the ligand's, which on the dev cohort
#: trades within-target for between-target exactly as the rotation recipe does.
_UNION_ALLATOMS_SEEDS = (34, 35, 36, 37, 38, 39, 40, 41)

UNION_ALLATOMS_VARIANTS: dict[str, Variant] = {
    _affinity_label_arm_name("e250_unionallatoms", seed): Variant(
        name=_affinity_label_arm_name("e250_unionallatoms", seed),
        description=(
            "e250 tokenizer + MLM, union corpus, all-atom readout, "
            f"training seed {seed}."
        ),
        vqvae=E250_VQ,
        mlm=E250_MLM,
        heads=(HeadSpec(f"aff_e250_unionallatoms_s{seed}", "kdki-mean"),),
        readout="allatoms",
        notes="fourth recipe of the mixing round: all-atom readout",
    )
    for seed in _UNION_ALLATOMS_SEEDS
}

UNION_ALLATOMS_SEED_FAMILIES: dict[str, dict[int, str]] = {
    "e250_unionallatoms": {
        seed: _affinity_label_arm_name("e250_unionallatoms", seed)
        for seed in _UNION_ALLATOMS_SEEDS
    }
}

#: The heavy recipe trained past the 20-epoch cap (2026-09-15), pre-registered
#: in docs/notes/2026-09-15_prereg_heavy_longer_training.md. Its heads peak at
#: epochs 13-18 and stop at 17-19 against that cap, while the CA recipes peak at
#: 3-9 -- the documents are ~1.5x longer, so the cap binds for this recipe only.
_UNION_HEAVY_LONG_SEEDS = (50, 51, 52, 53, 54, 55, 56, 57)

UNION_HEAVY_LONG_VARIANTS: dict[str, Variant] = {
    _affinity_label_arm_name("e250_unionheavy_long", seed): Variant(
        name=_affinity_label_arm_name("e250_unionheavy_long", seed),
        description=(
            "e250 tokenizer + MLM, union corpus with heavy-atom pocket selection, "
            f"40-epoch cap, training seed {seed}."
        ),
        vqvae=E250_VQ,
        mlm=E250_MLM,
        heads=(HeadSpec(f"aff_e250_unionheavylong_s{seed}", "kdki-mean"),),
        pocket_distance="heavy",
        notes="heavy recipe with --max-epochs 40; judged inside mixing ensembles",
    )
    for seed in _UNION_HEAVY_LONG_SEEDS
}

UNION_HEAVY_LONG_SEED_FAMILIES: dict[str, dict[int, str]] = {
    "e250_unionheavy_long": {
        seed: _affinity_label_arm_name("e250_unionheavy_long", seed)
        for seed in _UNION_HEAVY_LONG_SEEDS
    }
}

#: The three-recipe headline round (2026-09-15), pre-registered in
#: docs/notes/2026-09-15_prereg_three_recipe_headline.md before any of these
#: heads existed. The composition is what two PASSED pre-registrations imply --
#: plain + rotation (cross-recipe round) and + heavy pocket (third-recipe round)
#: -- in equal numbers, on seeds that have never been scored. Judged once.
#: (family, head stem, corpus blurb, pocket distance, seeds)
_V3_ARMS = (
    ("e250_unionnc_corr_v3", "aff_e250_unionnccorr",
     "union corpus, batch-correlation loss (0.7)", "ca",
     (58, 59, 60, 61, 62, 63, 64, 65)),
    ("e250_unionr8_ens_v3", "aff_e250_unionr8",
     "union corpus with 8 orientations per complex", "ca",
     (66, 67, 68, 69, 70, 71, 72, 73)),
    ("e250_unionheavy_v3", "aff_e250_unionheavy",
     "union corpus with heavy-atom pocket selection", "heavy",
     (74, 75, 76, 77, 78, 79, 80, 81)),
)

V3_VARIANTS: dict[str, Variant] = {
    _affinity_label_arm_name(family, seed): Variant(
        name=_affinity_label_arm_name(family, seed),
        description=f"e250 tokenizer + MLM, {corpus}, training seed {seed}.",
        vqvae=E250_VQ,
        mlm=E250_MLM,
        heads=(HeadSpec(f"{stem}_s{seed}", "kdki-mean"),),
        pocket_distance=pocket,
        notes="three-recipe headline round: fresh seeds, judged as the 24-model mix",
    )
    for family, stem, corpus, pocket, seeds in _V3_ARMS
    for seed in seeds
}

V3_SEED_FAMILIES: dict[str, dict[int, str]] = {
    family: {seed: _affinity_label_arm_name(family, seed) for seed in seeds}
    for family, _stem, _corpus, _pocket, seeds in _V3_ARMS
}

#: Reading every encoder layer instead of only the last (2026-09-15),
#: pre-registered in docs/notes/2026-09-15_prereg_layer_mix.md before the option
#: was implemented. Same corpus and recipe as its control e250_unionnc_corr_v3;
#: the only difference is a learned softmax mix over the embedding output and
#: all 13 layers, which is the first change here to the LAYER the head reads
#: rather than to which tokens it pools.
_LAYERMIX_SEEDS = (82, 83, 84, 85, 86, 87, 88, 89)

LAYERMIX_VARIANTS: dict[str, Variant] = {
    _affinity_label_arm_name("e250_unionnc_layermix", seed): Variant(
        name=_affinity_label_arm_name("e250_unionnc_layermix", seed),
        description=(
            "e250 tokenizer + MLM, union corpus, correlation loss, learned mix "
            f"over every encoder layer, training seed {seed}."
        ),
        vqvae=E250_VQ,
        mlm=E250_MLM,
        heads=(HeadSpec(f"aff_e250_unionnclayermix_s{seed}", "kdki-mean"),),
        layer_mix=True,
        notes="layer-mix round: softmax over the embedding output and all layers",
    )
    for seed in _LAYERMIX_SEEDS
}

LAYERMIX_SEED_FAMILIES: dict[str, dict[int, str]] = {
    "e250_unionnc_layermix": {
        seed: _affinity_label_arm_name("e250_unionnc_layermix", seed)
        for seed in _LAYERMIX_SEEDS
    }
}

#: The layer-mix round with weights that can actually learn (2026-09-15),
#: pre-registered in docs/notes/2026-09-15_prereg_layer_mix_learnable.md. The
#: first attempt left the 14 mixing scalars uniform after training, so it
#: measured an unweighted average of every layer and lost; this one starts the
#: mix on the last layer and gives those scalars their own learning rate.
#: Same corpus, recipe and control as that round.
_LAYERMIX_LR_SEEDS = (90, 91, 92, 93, 94, 95, 96, 97)

LAYERMIX_LR_VARIANTS: dict[str, Variant] = {
    _affinity_label_arm_name("e250_unionnc_layermix_lr", seed): Variant(
        name=_affinity_label_arm_name("e250_unionnc_layermix_lr", seed),
        description=(
            "e250 tokenizer + MLM, union corpus, correlation loss, learnable mix "
            f"over every encoder layer, training seed {seed}."
        ),
        vqvae=E250_VQ,
        mlm=E250_MLM,
        heads=(HeadSpec(f"aff_e250_unionnclayermixlr_s{seed}", "kdki-mean"),),
        layer_mix=True,
        notes="layer-mix round 2: last-layer init, dedicated LR for the mixing scalars",
    )
    for seed in _LAYERMIX_LR_SEEDS
}

LAYERMIX_LR_SEED_FAMILIES: dict[str, dict[int, str]] = {
    "e250_unionnc_layermix_lr": {
        seed: _affinity_label_arm_name("e250_unionnc_layermix_lr", seed)
        for seed in _LAYERMIX_LR_SEEDS
    }
}

#: Scaling the three-recipe mix from 24 to 36 models (2026-09-15),
#: pre-registered in docs/notes/2026-09-15_prereg_mix_scale36.md before these
#: heads existed. Four more seeds of each recipe; the 24 v3 models are reused,
#: so this is a scale-up of an already-scored mix, not a fresh replication.
#: (family, head stem, corpus blurb, pocket distance, seeds)
_V4_ARMS = (
    ("e250_unionnc_corr_v4", "aff_e250_unionnccorr",
     "union corpus, batch-correlation loss (0.7)", "ca",
     (98, 99, 100, 101)),
    ("e250_unionr8_ens_v4", "aff_e250_unionr8",
     "union corpus with 8 orientations per complex", "ca",
     (102, 103, 104, 105)),
    ("e250_unionheavy_v4", "aff_e250_unionheavy",
     "union corpus with heavy-atom pocket selection", "heavy",
     (106, 107, 108, 109)),
)

V4_VARIANTS: dict[str, Variant] = {
    _affinity_label_arm_name(family, seed): Variant(
        name=_affinity_label_arm_name(family, seed),
        description=f"e250 tokenizer + MLM, {corpus}, training seed {seed}.",
        vqvae=E250_VQ,
        mlm=E250_MLM,
        heads=(HeadSpec(f"{stem}_s{seed}", "kdki-mean"),),
        pocket_distance=pocket,
        notes="mix scale-up round: judged only as the 36-model mix",
    )
    for family, stem, corpus, pocket, seeds in _V4_ARMS
    for seed in seeds
}

V4_SEED_FAMILIES: dict[str, dict[int, str]] = {
    family: {seed: _affinity_label_arm_name(family, seed) for seed in seeds}
    for family, _stem, _corpus, _pocket, seeds in _V4_ARMS
}

#: How much encoder adaptation does this task want (2026-09-15), pre-registered
#: in docs/notes/2026-09-15_prereg_lora.md. Every other arm here fine-tunes the
#: whole ~105M encoder (freeze_encoder defaults to False and no head script
#: passes --freeze-encoder); these two constrain it instead -- head-only, and
#: head plus rank-4 LoRA adapters on each attention block.
#: (family, head stem, blurb, lora rank, seeds)
_ADAPT_ARMS = (
    ("e250_unionnc_frozen", "aff_e250_unionncfrozen",
     "frozen encoder, head only", 0,
     (110, 111, 112, 113, 114, 115, 116, 117)),
    ("e250_unionnc_lora", "aff_e250_unionnclora",
     "frozen encoder with rank-4 LoRA adapters", 4,
     (118, 119, 120, 121, 122, 123, 124, 125)),
)

ADAPT_VARIANTS: dict[str, Variant] = {
    _affinity_label_arm_name(family, seed): Variant(
        name=_affinity_label_arm_name(family, seed),
        description=(
            f"e250 tokenizer + MLM, union corpus, correlation loss, {blurb}, "
            f"training seed {seed}."
        ),
        vqvae=E250_VQ,
        mlm=E250_MLM,
        heads=(HeadSpec(f"{stem}_s{seed}", "kdki-mean"),),
        lora_rank=rank,
        notes="adaptation-budget round: judged against the full fine-tuning control",
    )
    for family, stem, blurb, rank, seeds in _ADAPT_ARMS
    for seed in seeds
}

ADAPT_SEED_FAMILIES: dict[str, dict[int, str]] = {
    family: {seed: _affinity_label_arm_name(family, seed) for seed in seeds}
    for family, _stem, _blurb, _rank, seeds in _ADAPT_ARMS
}

#: Distilling the 8-seed ensemble into single heads (2026-09-15),
#: pre-registered in docs/notes/2026-09-15_prereg_distillation.md. Same corpus
#: and recipe as the control e250_unionnc_corr_v3; the students additionally see
#: that family's mean prediction on each training document. Distillation is a
#: training-time term only, so the state_dict is unchanged and inference needs
#: no extra field.
_KD_SEEDS = (126, 127, 128, 129, 130, 131, 132, 133)

KD_VARIANTS: dict[str, Variant] = {
    _affinity_label_arm_name("e250_unionnc_kd", seed): Variant(
        name=_affinity_label_arm_name("e250_unionnc_kd", seed),
        description=(
            "e250 tokenizer + MLM, union corpus, correlation loss, distilled from "
            f"the 8-seed corr_v3 ensemble, training seed {seed}."
        ),
        vqvae=E250_VQ,
        mlm=E250_MLM,
        heads=(HeadSpec(f"aff_e250_unionnckd_s{seed}", "kdki-mean"),),
        notes="distillation round: judged against the teacher's own family",
    )
    for seed in _KD_SEEDS
}

KD_SEED_FAMILIES: dict[str, dict[int, str]] = {
    "e250_unionnc_kd": {
        seed: _affinity_label_arm_name("e250_unionnc_kd", seed) for seed in _KD_SEEDS
    }
}

#: Input masking during head fine-tuning (2026-09-16), pre-registered in
#: docs/notes/2026-09-16_prereg_input_masking.md before any head existed. Same
#: corpus, driver and recipe as the control e250_unionnc_corr_v3, on the SAME
#: seeds (58-65), so the comparison is paired; the three added flags are
#: --input-mask-prob 0.15 --input-mask-side ligand --input-mask-pool keep.
#:
#: Like distillation this is a TRAIN-time term only: 15% of the train split's
#: ligand tokens become the encoder's <mask> id, validation is never masked and
#: inference never masks. The state_dict and the scoring path are unchanged, so
#: unlike readout / layer_mix / lora_rank this needs no field here to keep
#: training and scoring in step -- there is no second path to get wrong.
#:
#: The claim being tested is per-model quality, which is where the whole R
#: deficit sits (ours 0.7920 single against GenScore's 0.8127). The nearest
#: precedent, rotation sampling, reduced memorisation and still bought only
#: family-ensemble gains, so the pre-registration puts its primary rule on the
#: single-seed mean and treats a family-only gain as a negative.
_IMASK_SEEDS = (58, 59, 60, 61, 62, 63, 64, 65)

IMASK_VARIANTS: dict[str, Variant] = {
    _affinity_label_arm_name("e250_unionnc_imask", seed): Variant(
        name=_affinity_label_arm_name("e250_unionnc_imask", seed),
        description=(
            "e250 tokenizer + MLM, union corpus, correlation loss, 15% of the "
            f"training ligand tokens masked, training seed {seed}."
        ),
        vqvae=E250_VQ,
        mlm=E250_MLM,
        heads=(HeadSpec(f"aff_e250_unionncimask_s{seed}", "kdki-mean"),),
        notes="input-masking round: train-time only; scored by the unmasked path",
    )
    for seed in _IMASK_SEEDS
}

IMASK_SEED_FAMILIES: dict[str, dict[int, str]] = {
    "e250_unionnc_imask": {
        seed: _affinity_label_arm_name("e250_unionnc_imask", seed)
        for seed in _IMASK_SEEDS
    }
}

#: The `drop` follow-up (2026-09-16), pre-registered in
#: docs/notes/2026-09-16_prereg_input_masking_drop.md after `keep` was refuted.
#:
#: `keep` left a masked ligand row inside the pooled mean, so training averaged a
#: mixture of observed and inferred rows while inference averages observed rows
#: only; it lost dR -0.0072 / drho -0.0263 with between up and within down.
#: `drop` takes the masked position out of the mean, so the pooled statistic is
#: the same quantity in training and at inference.
#:
#: A FRESH CONTROL is trained at the same new seeds. `e250_unionnc_corr_v3` has
#: already been read as the `keep` round's control, so judging a second treatment
#: against it would be a second test against a realisation already seen, and
#: `n_better` would no longer be a fresh statistic.
_IMASK_DROP_SEEDS = (134, 135, 136, 137, 138, 139, 140, 141)

_IMASK_DROP_ARMS: tuple[tuple[str, str, str], ...] = (
    ("e250_unionnc_imaskdrop", "aff_e250_unionncimdrop",
     "15% of the training ligand tokens masked and dropped from the pooled mean"),
    ("e250_unionnc_ctl_v5", "aff_e250_unionncctl5",
     "no masking -- the seed-matched control for the drop round"),
)

IMASK_DROP_VARIANTS: dict[str, Variant] = {
    _affinity_label_arm_name(family, seed): Variant(
        name=_affinity_label_arm_name(family, seed),
        description=(
            f"e250 tokenizer + MLM, union corpus, correlation loss, {blurb}, "
            f"training seed {seed}."
        ),
        vqvae=E250_VQ,
        mlm=E250_MLM,
        heads=(HeadSpec(f"{stem}_s{seed}", "kdki-mean"),),
        notes="drop round: train-time only; scored by the unmasked path",
    )
    for family, stem, blurb in _IMASK_DROP_ARMS
    for seed in _IMASK_DROP_SEEDS
}

IMASK_DROP_SEED_FAMILIES: dict[str, dict[int, str]] = {
    family: {
        seed: _affinity_label_arm_name(family, seed) for seed in _IMASK_DROP_SEEDS
    }
    for family, _stem, _blurb in _IMASK_DROP_ARMS
}

#: The correlation-loss round (2026-09-15), pre-registered in
#: docs/notes/2026-09-15_prereg_correlation_loss.md before the arm was trained.
#:
#: Read out of the baselines' source: OnionNet-2 trains on
#: alpha*(1-PCC) + (1-alpha)*RMSE with alpha 0.7, and GenScore adds a corrcoef
#: term to its loss. Both of the models that beat us on scoring power optimise
#: the correlation directly, while we optimise per-complex smooth-L1 and then
#: report a correlation. The weight is copied from OnionNet-2's default, not
#: swept. The control is e250_union_noconf: same corpus, same driver, same
#: recipe, already trained and already scored.
#: 7-13 are the pre-registered correlation-loss round; 14-17 were added on
#: 2026-09-15 to measure ensemble size on the best arm (see
#: docs/notes/2026-09-15_prereg_ensemble_size.md), same recipe.
_CORR_LOSS_SEEDS = (7, 11, 12, 13, 14, 15, 16, 17)

CORR_LOSS_VARIANTS: dict[str, Variant] = {
    _affinity_label_arm_name("e250_unionnc_corr", seed): Variant(
        name=_affinity_label_arm_name("e250_unionnc_corr", seed),
        description=(
            "e250 tokenizer + MLM, union_noconf corpus, loss "
            f"0.3*smooth_l1 + 0.7*(1-PCC), training seed {seed}."
        ),
        vqvae=E250_VQ,
        mlm=E250_MLM,
        heads=(HeadSpec(f"aff_e250_unionnccorr_s{seed}", "kdki-mean"),),
        notes="correlation-loss arm: differs from its control only in --corr-weight",
    )
    for seed in _CORR_LOSS_SEEDS
}

CORR_LOSS_SEED_FAMILIES: dict[str, dict[int, str]] = {
    "e250_unionnc_corr": {
        seed: _affinity_label_arm_name("e250_unionnc_corr", seed)
        for seed in _CORR_LOSS_SEEDS
    }
}

#: The group-mean-loss round (2026-09-15), pre-registered in
#: docs/notes/2026-09-15_prereg_group_mean_loss.md before any head was trained.
#:
#: The stratified analysis put the deficit at target level: cluster-mean error
#: against cluster-mean pK has slope -0.190 for us and -0.048 for GIGN, i.e. we
#: compress target-level differences by ~19%. Per-complex smooth-L1 cannot see
#: that shape of error, so the treatment adds a term on each protein group's
#: MEAN residual. Both arms train on lm_tokens_e250aff_unionnc_grp (union_noconf
#: with real UniProt groups written into .grp; tokens byte-identical) and both
#: use group batching, so the only difference is the weight.
_GROUP_MEAN_ARMS: dict[str, tuple[str, str]] = {
    # arm family -> (head run-name stem, what the arm is)
    "e250_unionnc_gm": ("aff_e250_unionncgm", "group-mean loss weight 1.0"),
    # Amendment 1: this one trained on RANDOM batches (see below).
    "e250_unionnc_gm_ctl": ("aff_e250_unionncgmctl", "weight 0, random batching"),
    # The control the verdict uses: grouped batching like the treatment,
    # weight 0. The first control passed --group-batch-partition but the
    # loader only switched to group batches when a grouped LOSS was on, so
    # it trained on random batches (479 steps/epoch against 722).
    "e250_unionnc_gm_ctl2": ("aff_e250_unionncgmctl2", "grouped batching, weight 0"),
}

GROUP_MEAN_VARIANTS: dict[str, Variant] = {
    _affinity_label_arm_name(fam, seed): Variant(
        name=_affinity_label_arm_name(fam, seed),
        description=(
            f"e250 tokenizer + MLM, union corpus with UniProt groups, {desc}, "
            f"training seed {seed}."
        ),
        vqvae=E250_VQ,
        mlm=E250_MLM,
        heads=(HeadSpec(f"{stem}_s{seed}", "kdki-mean"),),
        notes="group-mean pair: the arms differ only in --group-mean-weight",
    )
    for fam, (stem, desc) in _GROUP_MEAN_ARMS.items()
    for seed in _AFFINITY_LABEL_SEEDS
}

GROUP_MEAN_SEED_FAMILIES: dict[str, dict[int, str]] = {
    fam: {seed: _affinity_label_arm_name(fam, seed) for seed in _AFFINITY_LABEL_SEEDS}
    for fam in _GROUP_MEAN_ARMS
}

#: The rotation-sampling round (2026-09-15), pre-registered in
#: docs/notes/2026-09-15_prereg_rotation_sampling.md before any head was trained.
#:
#: Training sees one fixed orientation per complex while inference averages
#: 16-32 frames, and tokens are frame-dependent. e250_rot8 trained on the
#: 8-rotation corpus as 8x the documents and memorised it; here the treatment
#: indexes one entry per complex and draws one of its 8 orientations per
#: access (``--rotations-per-complex 8``). Both arms are trained by the same
#: driver, so seeds pair; the old e250_rot1 family (another driver) is not the
#: control. BioLiP-only corpora: this round decides the lever, not a headline.
_ROTATION_SAMPLING_ARMS: dict[str, tuple[str, str]] = {
    # arm family -> (head run-name stem, corpus description)
    "e250_rot1_ctl": ("aff_e250_rot1ctl", "1 orientation per complex"),
    "e250_rot8_sampled": ("aff_e250_rot8smp", "8 orientations, one drawn per access"),
}

ROTATION_SAMPLING_VARIANTS: dict[str, Variant] = {
    _affinity_label_arm_name(fam, seed): Variant(
        name=_affinity_label_arm_name(fam, seed),
        description=(
            f"e250 tokenizer + MLM, BioLiP KD/KI corpus, {desc}, "
            f"training seed {seed}."
        ),
        vqvae=E250_VQ,
        mlm=E250_MLM,
        heads=(HeadSpec(f"{stem}_s{seed}", "kdki-mean"),),
        notes="rotation-sampling pair: the arms differ only in --rotations-per-complex",
    )
    for fam, (stem, desc) in _ROTATION_SAMPLING_ARMS.items()
    for seed in _AFFINITY_LABEL_SEEDS
}

ROTATION_SAMPLING_SEED_FAMILIES: dict[str, dict[int, str]] = {
    fam: {seed: _affinity_label_arm_name(fam, seed) for seed in _AFFINITY_LABEL_SEEDS}
    for fam in _ROTATION_SAMPLING_ARMS
}

#: The concat-readout round (2026-09-14 ~20:40), pre-registered in
#: docs/notes/2026-09-14_prereg_concat_readout.md before any head was trained.
#:
#: On CASF-2016 the R deficit to GIGN / OnionNet-2 is almost entirely the
#: BETWEEN-target part (which protein binds strongly on average); within targets
#: we are level. The ligand readout loses the between-target part, pooling all
#: atoms recovers it but dilutes the within-target part, and averaging the two
#: means ("balanced") washes the pocket out again. "concat" keeps the pocket
#: mean and ligand mean side by side. Two corpora, each paired against its own
#: ligand-readout control under the same driver, so seed numbers pair
#: trajectories.
_CONCAT_ARMS: dict[str, str] = {
    # family -> head run-name stem
    "e250_unionnc_concat": "aff_e250_unionnc_concat",
    "e250_kdkictl_concat": "aff_e250_kdkictl_concat",
}
_CONCAT_SEEDS = (7, 11, 12, 13)


def _concat_arm_name(family: str, seed: int) -> str:
    """Seed 7 keeps the bare family name, as in every other family here."""
    if seed == _BARE_NAME_SEED:
        return family
    return f"{family}_s{seed}"


CONCAT_READOUT_VARIANTS: dict[str, Variant] = {
    _concat_arm_name(fam, seed): Variant(
        name=_concat_arm_name(fam, seed),
        description=(
            "e250 tokenizer + MLM, "
            f"{'union_noconf' if 'unionnc' in fam else 'KD/KI control'} corpus, "
            f"concat readout (pocket mean | ligand mean), training seed {seed}."
        ),
        vqvae=E250_VQ,
        mlm=E250_MLM,
        heads=(HeadSpec(f"{stem}_s{seed}", "kdki-mean"),),
        readout="concat",
        notes="concat-readout round; identical to its control except --readout concat",
    )
    for fam, stem in _CONCAT_ARMS.items()
    for seed in _CONCAT_SEEDS
}

CONCAT_READOUT_SEED_FAMILIES: dict[str, dict[int, str]] = {
    fam: {seed: _concat_arm_name(fam, seed) for seed in _CONCAT_SEEDS}
    for fam in _CONCAT_ARMS
}

_UNIONNC_READOUT_ARMS: dict[str, str] = {"allatoms": "all", "balanced": "bal"}
_UNIONNC_SEEDS = (7, 11, 12, 13)

def _unionnc_arm_name(readout: str, seed: int) -> str:
    """Seed 7 keeps the bare family name, as in every other family here."""
    if seed == _BARE_NAME_SEED:
        return f"e250_unionnc_{readout}"
    return f"e250_unionnc_{readout}_s{seed}"


UNIONNC_READOUT_VARIANTS: dict[str, Variant] = {
    _unionnc_arm_name(readout, seed): Variant(
        name=_unionnc_arm_name(readout, seed),
        description=(
            f"e250 + the union_noconf corpus read with {readout} pooling, "
            f"training seed {seed}."
        ),
        vqvae=E250_VQ,
        mlm=E250_MLM,
        # A run name, not a path: resolve_scoring_head picks that run's
        # lowest-val-loss checkpoint, so the arm is registerable before the
        # training job has produced a file.
        heads=(HeadSpec(f"aff_e250_unionnc_{tag}_s{seed}", "kdki-mean"),),
        readout=readout,
        notes=(
            "union_noconf leads on R, the readout leads on rho; this is whether "
            "they combine. Same 151 contradictory-label exclusions as its control."
        ),
    )
    for readout, tag in _UNIONNC_READOUT_ARMS.items()
    for seed in _UNIONNC_SEEDS
}

UNIONNC_READOUT_SEED_FAMILIES: dict[str, dict[int, str]] = {
    f"e250_unionnc_{readout}": {
        seed: _unionnc_arm_name(readout, seed) for seed in _UNIONNC_SEEDS
    }
    for readout in _UNIONNC_READOUT_ARMS
}

#: Seeds 13-15 of the two families whose ensembles sit on the frontier.
#:
#: Nothing about the recipe changes -- same corpus, same MLM, same head, same
#: flags, a different seed. The reason is variance, not mechanism: z-summing
#: all seeds of a family is worth +0.010..+0.029 R and +0.004..+0.047 rho over
#: its mean single seed, and that gain grows with member count. At three seeds
#: each, the zero-selection all-families ensemble lands 0.0019 under GenScore on
#: R and 0.0053 under on rho.
#:
#: Adding seeds cannot be selection on the test set: every seed of a family
#: enters its ensemble, and which seeds exist was fixed before any was scored.
_EXTRA_SEEDS = (13, 14, 15)
#: arm family -> (its run-name stem, the corpus it trains on)
_EXTRA_SEED_FAMILIES: dict[str, str] = {
    "e250_rot1": "aff_e250_r1",
    "e250_union_noconf": "aff_e250_unionnc",
}

EXTRA_SEED_VARIANTS: dict[str, Variant] = {
    f"{fam}_s{seed}": Variant(
        name=f"{fam}_s{seed}",
        description=f"{fam} retrained with seed {seed}.",
        vqvae=E250_VQ,
        mlm=E250_MLM,
        heads=(HeadSpec(f"{stem}_s{seed}", "kdki-mean"),),
        notes="added to deepen the family ensemble; recipe identical to its seed 7",
    )
    for fam, stem in _EXTRA_SEED_FAMILIES.items()
    for seed in _EXTRA_SEEDS
}

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
} | (
    DEV_INPUT_ARMS
    | DEV_READOUT_ARM_VARIANTS
    | UNIONNC_READOUT_VARIANTS
    | EXTRA_SEED_VARIANTS
    | BACKBONE_SWAP_VARIANTS
    | AFFINITY_LABEL_VARIANTS
    | CONCAT_READOUT_VARIANTS
    | UNION_LABEL_VARIANTS
    | ROTATION_SAMPLING_VARIANTS
    | GROUP_MEAN_VARIANTS
    | CORR_LOSS_VARIANTS
    | UNION_R8_VARIANTS
    | CROSSMIX_VARIANTS
    | UNION_HEAVY_VARIANTS
    | UNION_ALLATOMS_VARIANTS
    | UNION_HEAVY_LONG_VARIANTS
    | V3_VARIANTS
    | LAYERMIX_VARIANTS
    | LAYERMIX_LR_VARIANTS
    | V4_VARIANTS
    | ADAPT_VARIANTS
    | KD_VARIANTS
    | IMASK_VARIANTS
    | IMASK_DROP_VARIANTS
)

#: The ESM3 x ConfSeq baseline on the three headline recipes (2026-09-19).
#: Same corpora (the stapled tokenization of the same union complexes, the same
#: val proteins), same flags and the SAME seeds as the V3 families, so each
#: stapled head pairs with the ProLIT head of its seed; only the tokenizer and
#: the MLM it implies (``mlm_stapled``, the rescoring baseline's backbone)
#: differ. Scored in one frame: ESM3 and ConfSeq codes do not rotate.
STAPLED_MLM = "pocket-ligand-mlm/mlm_stapled/checkpoints/mlm-e02-vl0.7394.ckpt"
_STAPLED_ARMS = (
    # (family, head stem, ProLIT twin family, pocket distance)
    ("stapled_unionnc_corr", "aff_stapled_unionnccorr", "e250_unionnc_corr_v3", "ca"),
    ("stapled_unionr8_ens", "aff_stapled_unionr8", "e250_unionr8_ens_v3", "ca"),
    ("stapled_unionheavy", "aff_stapled_unionheavy", "e250_unionheavy_v3", "heavy"),
)

STAPLED_VARIANTS: dict[str, Variant] = {
    f"{family}_s{seed}": Variant(
        name=f"{family}_s{seed}",
        description=f"ESM3 x ConfSeq + mlm_stapled, {twin} recipe, seed {seed}.",
        mlm=STAPLED_MLM,
        heads=(HeadSpec(f"{stem}_s{seed}", "kdki-mean"),),
        codebook_size=12733,
        pocket_distance=pocket,
        stapled=StapledSpec(),
        notes=f"stapled baseline; seed-paired with {twin}_s{seed}",
    )
    for family, stem, twin, pocket in _STAPLED_ARMS
    for seed in V3_SEED_FAMILIES[twin]
}
REGISTRY.update(STAPLED_VARIANTS)

#: Stapled family -> the ProLIT family it is seed-paired with.
STAPLED_TWIN: dict[str, str] = {family: twin for family, _s, twin, _p in _STAPLED_ARMS}
STAPLED_SEED_FAMILIES: dict[str, dict[int, str]] = {
    family: {seed: f"{family}_s{seed}" for seed in V3_SEED_FAMILIES[twin]}
    for family, _s, twin, _p in _STAPLED_ARMS
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
    "e250_rot1": {
        7: "e250_rot1",
        11: "e250_rot1_s11",
        12: "e250_rot1_s12",
        13: "e250_rot1_s13",
        14: "e250_rot1_s14",
        15: "e250_rot1_s15",
    },
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
        13: "e250_union_noconf_s13",
        14: "e250_union_noconf_s14",
        15: "e250_union_noconf_s15",
    },
} | (
    DEV_SEED_FAMILIES
    | DEV_READOUT_SEED_FAMILIES
    | UNIONNC_READOUT_SEED_FAMILIES
    | BACKBONE_SWAP_SEED_FAMILIES
    | AFFINITY_LABEL_SEED_FAMILIES
    | CONCAT_READOUT_SEED_FAMILIES
    | UNION_LABEL_SEED_FAMILIES
    | ROTATION_SAMPLING_SEED_FAMILIES
    | GROUP_MEAN_SEED_FAMILIES
    | CORR_LOSS_SEED_FAMILIES
    | UNION_R8_SEED_FAMILIES
    | CROSSMIX_SEED_FAMILIES
    | UNION_HEAVY_SEED_FAMILIES
    | UNION_ALLATOMS_SEED_FAMILIES
    | UNION_HEAVY_LONG_SEED_FAMILIES
    | V3_SEED_FAMILIES
    | LAYERMIX_SEED_FAMILIES
    | LAYERMIX_LR_SEED_FAMILIES
    | V4_SEED_FAMILIES
    | ADAPT_SEED_FAMILIES
    | KD_SEED_FAMILIES
    | IMASK_SEED_FAMILIES
    | IMASK_DROP_SEED_FAMILIES
)

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
    # The dev round's own control, not e250_rot1: these arms changed the corpus
    # (7,714 sites, one frame) as well as the inputs, so against e250_rot1 an
    # input repair would be credited with a corpus change too.
    **{
        f"e250_dev_{arm}": "e250_dev_ca_legacy"
        for arm in _DEV_ROUND_ARMS
        if arm != "ca_legacy"
    },
    # The readout arms share the ca_legacy corpus and differ from it by pooling
    # alone, so it is their control too.
    **{f"e250_dev_{arm}": "e250_dev_ca_legacy" for arm in _DEV_READOUT_ARMS},
    # Same corpus, ligand readout: the control isolates the pooling change.
    **{
        f"e250_unionnc_{readout}": "e250_union_noconf"
        for readout in _UNIONNC_READOUT_ARMS
    },
    # Same corpus, same ligand readout, same recipe: the control isolates the
    # backbone, which is the only flag that differs.
    "e250_unionnc_lo": "e250_union_noconf",
    # Same builder, same tokenizer, same exclusions, same day: the control
    # isolates --affinity-types, which is the only flag that differs.
    "e250_aff_ic50": "e250_aff_kdki_ctl",
    # Same corpus, same driver, ligand readout: isolates the readout.
    "e250_unionnc_concat": "e250_union_noconf",
    "e250_kdkictl_concat": "e250_aff_kdki_ctl",
    # Same builders, same tokenizer, same exclusion rule, same recipe: the
    # control isolates --affinity-types on both halves of the union.
    "e250_unionic50": "e250_unionkdki",
    # Same driver, same val file (byte-identical), same recipe: the control
    # isolates --rotations-per-complex.
    "e250_rot8_sampled": "e250_rot1_ctl",
    # Same corpus, same group batching, same recipe: the control isolates
    # --group-mean-weight.
    # Same corpus, same driver, same recipe, already scored: the control
    # isolates --corr-weight.
    "e250_unionnc_corr": "e250_union_noconf",
    # Same corpus plus rotations, same driver and recipe: judged as a family.
    "e250_unionr8_ens": "e250_union_noconf",
    # Fresh seeds of the same two recipes, judged as a mix against each other.
    "e250_unionnc_corr_v2": "e250_union_noconf",
    "e250_unionr8_ens_v2": "e250_union_noconf",
    # Third recipe: judged only inside mixing ensembles, never seed-paired.
    "e250_unionheavy": "e250_union_noconf",
    # Fourth recipe: judged only inside mixing ensembles, never seed-paired.
    "e250_unionallatoms": "e250_union_noconf",
    # Longer-trained heavy recipe: judged against the standard heavy seeds.
    "e250_unionheavy_long": "e250_unionheavy",
    # Same corpus, same driver, same recipe, SAME seeds: the control isolates
    # --input-mask-prob. Paired seed by seed, unlike the mix-only v3 families.
    "e250_unionnc_imask": "e250_unionnc_corr_v3",
    # The drop round trains its OWN control at the same new seeds, because the
    # v3 control has already been read as the keep round's control.
    "e250_unionnc_imaskdrop": "e250_unionnc_ctl_v5",
    # Headline round: these are judged only as the 24-model mix, never seed-paired.
    "e250_unionnc_corr_v3": "e250_union_noconf",
    "e250_unionr8_ens_v3": "e250_union_noconf",
    "e250_unionheavy_v3": "e250_unionheavy",
    # Layer-mix: same corpus and recipe as its control, differing only in
    # which layers the head reads.
    "e250_unionnc_layermix": "e250_unionnc_corr_v3",
    "e250_unionnc_layermix_lr": "e250_unionnc_corr_v3",
    # Scale-up seeds: judged only inside the 36-model mix.
    "e250_unionnc_corr_v4": "e250_union_noconf",
    "e250_unionr8_ens_v4": "e250_union_noconf",
    "e250_unionheavy_v4": "e250_unionheavy",
    # Adaptation-budget round: both judged against the full fine-tuning arm.
    "e250_unionnc_frozen": "e250_unionnc_corr_v3",
    "e250_unionnc_lora": "e250_unionnc_corr_v3",
    # Distillation: judged against the family that taught it.
    "e250_unionnc_kd": "e250_unionnc_corr_v3",
    "e250_unionnc_gm": "e250_unionnc_gm_ctl2",
    # Kept so the mis-batched arm still pairs against the real control:
    # their difference is the batching, not the loss.
    "e250_unionnc_gm_ctl": "e250_unionnc_gm_ctl2",
}


#: Order the comparison table lists our arms in.
ARM_ORDER: tuple[str, ...] = (
    "e250_unionnc_allatoms",
    "e250_unionnc_balanced",
    "e250_dev_allatoms",
    "e250_dev_balanced",
    "e250_dev_heavy_validated",
    "e250_dev_heavy_legacy",
    "e250_dev_ca_validated",
    "e250_dev_ca_legacy",
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
    "E250_MLM_LO",
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

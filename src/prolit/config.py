from dataclasses import dataclass, field
from pathlib import Path

from prolit.seeding import DEFAULT_SEED
from prolit.tokenizers.descriptor_schema import ATOM_DESCRIPTOR_DIM


@dataclass
class HubDatasetConfig:
    """Config for loading CrossDocked2020 from HuggingFace Hub."""

    repo_id: str = "sakano/crossdocked2020"
    cache_dir: Path = Path("data/hub_cache")
    # Selects the manifest column ``{source_type}_fold{fold}`` used to assign
    # the official CrossDocked2020 train/test split when ``_setup_from_shards``
    # builds the descriptor split.
    fold: int = 0
    source_types: list[str] = field(default_factory=lambda: ["cdonly"])
    revision: str | None = None
    # When True, keep only ``label == 1`` (native-like / good) poses at manifest
    # load. CrossDocked ``label == 0`` poses are decoys (RMSD > 2 Å) built as
    # classifier negatives; training a *generative* model on them hurts molecule
    # fitness. Used by the all-atom pipeline.
    good_poses_only: bool = False
    # When True, drop from train/val every complex whose receptor PDB id appears
    # in an evaluation set that is NOT already handled by the fold split --
    # the CASF-2016 core set and the sbdd-bench targets. The fold's own test
    # side is excluded regardless, so it is not repeated here.
    #
    # This is off by default because turning it on changes which complexes a run
    # trains on, and the published runs were trained without it: 169 of the 285
    # CASF core-set entries sit on the CrossDocked *train* side of fold 0, so
    # those runs saw them. New runs should set it.
    exclude_eval_pdbs: bool = False
    # Where the CASF core-set id list lives. Relative to the repository root.
    casf_pdb_list: Path = Path("data/casf2016_pdbs.txt")


@dataclass
class CrossDockedConfig:
    data_dir: Path = Path("data")
    base_url: str = "http://bits.csb.pitt.edu/files/crossdock2020"
    data_tarball: str = "CrossDocked2020_v1.3.tgz"
    types_tarball: str = "CrossDocked2020_v1.3_types.tgz"
    batch_size: int = 32
    num_workers: int = 4
    test_size: float = 0.1
    val_size: float = 0.1
    random_state: int = 42
    max_pairs: int | None = None


@dataclass
class PocketExtractionConfig:
    """Config for extracting pocket residues around a ligand."""

    distance_cutoff: float = 8.0
    max_residues: int = 128
    #: Order of pocket residues in the token stream. "sequence" (default) is
    #: chain/residue order and reproduces every existing checkpoint. "distance"
    #: puts the residues nearest the ligand LAST, adjacent to the ligand block,
    #: so RoPE's decay lines up with spatial proximity. Changing this changes
    #: the token stream, so it invalidates LMs trained under the other setting.
    pocket_order: str = "sequence"
    #: Order the ligand's heavy atoms are emitted in. "file" is whatever the
    #: SDF stored -- what every existing checkpoint was trained on --
    #: and "buried_first" walks the bond graph outward from the pocket.
    #: Changing it changes the token stream and invalidates checkpoints.
    atom_order: str = "file"

    #: Widen each LIGAND atom's knn SEARCH set to the pocket, so its empty
    #: neighbour slots are filled by the nearest protein atoms. Descriptor width
    #: is unchanged (33-D); False reproduces the original descriptor exactly.
    #: Lives here rather than on the training config so it travels with the
    #: pocket settings into the tar-streaming workers, which receive only
    #: ``asdict(pocket_config)``.
    pocket_context: bool = False


def _default_atom_recon_weights() -> dict[str, float]:
    """Recon weights for the unified all-atom VQ-VAE (protein + ligand).

    ``coord`` + the six chemistry heads are trained on every atom; ``aa`` /
    ``bb_sc`` only on protein rows; ``clash`` only on ligand rows (all handled
    by per-source masking in :meth:`TransformerVQVAE._compute_recon_loss`).
    """
    return {
        "coord": 1.0,
        "element": 0.5,
        "charge": 0.1,
        "hybrid": 0.1,
        "aromatic": 0.1,
        "ring": 0.1,
        "numH": 0.1,
        "aa": 0.5,
        "bb_sc": 0.1,
        "clash": 5.0,
        # Only consumed when ``AtomVQVAEConfig.predict_knn_offsets`` is set.
        "knn_offsets": 1.0,
        # Only consumed when ``AtomVQVAEConfig.bond_distance_loss`` is set.
        # Sized so each starts at roughly a fifth of ``coord``: the earlier
        # attempt failed partly because the clash term sat at 5.0 x 0.007, an
        # eighth of coord's contribution, and so never steered anything.
        "bond12": 5.0,
        "bond13": 2.0,
        # Only consumed when ``AtomVQVAEConfig.distance_map_loss`` is set. 1.0,
        # the same as ``coord``: it is the same quantity -- where the atoms are
        # -- expressed as distances instead of positions, so there is no reason
        # to prefer one over the other.
        "dmap": 1.0,
        # Only consumed when ``AtomVQVAEConfig.pair_distance_loss`` is set. 1.0,
        # like ``coord``: one says where an atom is, the other what shape the
        # atoms make, and there is no ground for preferring either. ``coord``
        # cannot be dropped in its favour -- distances are invariant to rotation
        # and translation, so nothing else fixes the pose in the pocket frame.
        "pair": 1.0,
        # Only consumed when ``AtomVQVAEConfig.local_distance_loss`` is set.
        # 1.0, like ``coord`` and ``dmap``: three views of where the atoms are,
        # at three scales, with no ground for ranking them.
        "local": 1.0,
    }


@dataclass
class AtomVQVAEConfig:
    """Config for the unified all-atom VQ-VAE (one codebook, protein + ligand).

    ``max_seq_len`` sizes the positional-encoding buffer and must exceed the
    longest single atom sequence (a whole pocket, up to ~max_residues * heavy
    atoms/residue). 1024 covers an 8 Å pocket comfortably; lower
    ``max_residues`` or raise this if a pocket ever exceeds it.
    """

    descriptor_dim: int = ATOM_DESCRIPTOR_DIM
    hidden_dim: int = 256
    latent_dim: int = 16
    codebook_size: int = 8192
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    num_transformer_layers: int = 4
    num_attention_heads: int = 8
    transformer_feedforward_dim: int = 512
    transformer_dropout: float = 0.1
    max_seq_len: int = 1024
    domain: str = "atom"
    categorical_embed_dim: int = 8
    recon_weights: dict[str, float] = field(
        default_factory=_default_atom_recon_weights,
    )

    # --- local-geometry supervision (R1) --------------------------------
    #
    # The descriptor carries K nearest-neighbour spherical displacements as an
    # ENCODER INPUT but has no matching decoder head, so nothing ever asked the
    # decoder to place an atom correctly *relative to its neighbours*. Measured
    # consequence: the error vectors of two bonded atoms have cosine +0.37, i.e.
    # they are nearly independent, and bond-length MAE stays at 0.15 A even with
    # quantization switched off entirely.
    #
    # Adding the head creates parameters, so it is opt-in: a checkpoint trained
    # without it has no such weights and would fail a strict load.
    predict_knn_offsets: bool = False

    # Penalise the 1-2 and 1-3 distances OF THE DECODED COORDINATES against the
    # reference. Unlike ``predict_knn_offsets`` this constrains the coordinate
    # head itself rather than adding a parallel output, which is why the first
    # attempt at local-geometry supervision did nothing (see the 2026-08-04
    # note). The bond graph is derived in the loss from the reference geometry
    # by covalent radius, so no stored connectivity and no cache change.
    bond_distance_loss: bool = False
    #: Weight each categorical head's atoms by the inverse frequency of their
    #: own class. Off by default because every existing checkpoint was trained
    #: without it, and the constraint targets are calibrated to that scale.
    balanced_chem_loss: bool = False
    #: Train the decoder only: the encoder and the codebook keep the weights
    #: they were loaded with, so the codes -- and every token stream and
    #: language model built on them -- are unchanged.
    freeze_encoder: bool = False

    # Apply the 1-2 / 1-3 terms to PROTEIN atoms as well as ligand ones. They
    # are ligand-only by default, which is measurably a mistake: lDDT is itself
    # a local-distance score, and the arm that added the ligand-only term
    # (joint_bond) came out BELOW the arm without it on protein backbone
    # (TM 0.809 vs 0.826, lDDT 0.914 vs 0.925) -- the term pulled capacity to
    # the ligand and gave the protein nothing back. Backbone geometry is more
    # regular than a ligand's (N-CA 1.46, CA-C 1.52, C-N 1.33 A), so there is
    # less to learn, not more.
    #
    # The clash term stays ligand-only either way: it forbids inventing contacts
    # the crystal does not have, which is a generative failure mode, and a
    # pocket's packing is dense enough that the same floor would fight real
    # structure.
    bond_distance_all_sources: bool = False

    # Squared error on EVERY pairwise distance the reference keeps under
    # ``distance_map_cutoff``, for protein atoms.
    #
    # Added because extending the bonded terms to the protein did nothing for
    # backbone lDDT, and the arithmetic says why: a C-C bond is called at
    # 0.76 + 0.76 + 0.4 = 1.92 A, while the CA-CA distances backbone lDDT
    # actually scores sit at ~3.8 A. The 1-2 and 1-3 sets never contain a single
    # pair that metric looks at. Run vq_pbond scored 3/8 against the rival set
    # (TM 0.801, lDDT 0.911) -- below the arm without it.
    #
    # This term is on the same quantity at the scale that is measured. It is not
    # metric-gaming: preserving local distances is what structural quality
    # means, and unlike the per-atom coord loss it is frame-independent, so it
    # cannot be satisfied by getting the canonical frame right and the shape
    # wrong.
    # One pairwise term in place of four. ``bond12``/``bond13``/``clash``/
    # ``dmap`` differ only in which distances they look at and how hard they
    # push, and the pushing was done with hand-set weights (5.0 / 2.0 / 5.0 /
    # 1.0) that also forced the per-source split: the short-range terms ran on
    # ligand rows and the long-range one on protein rows.
    #
    # Measuring the error RELATIVE to the reference distance removes the need
    # for any of that. The same 0.1 A slip is a 7% error on a bond and a 0.7%
    # error on a 14 A pair, so proximity is weighted by the form of the
    # expression rather than by a constant. Measured on 237 real pockets and
    # ligands, bonded pairs take 0.0085 of an unweighted absolute loss, 0.0371
    # of the current hand-weighted one, and 0.0770 of this -- the emphasis the
    # 5.0 was buying, arrived at without the 5.0.
    #
    # The clash floor goes too: putting a pair the crystal holds at 5 A onto
    # 1 A is a relative error of 0.8, which this already punishes far harder
    # than the hinge did. ``keep_clash`` exists to test that claim rather than
    # assume it.
    # Local geometry as ONE term instead of two. ``bond12`` and ``bond13``
    # differ only in which pairs they cover (1-2 and 1-3) and how hard they push
    # (5.0 and 2.0); scoring their union with a relative error removes the
    # ratio, because a 1-2 pair at 1.5 A and a 1-3 pair at 2.5 A already differ
    # in how much the same absolute slip costs.
    #
    # This keeps the local / long-range split that ``pair_distance_loss``
    # removed, which was measured to be necessary: collapsing everything into
    # one term over 15 A took bond MAE from 0.075 to 0.233 A and PB-valid from
    # 0.685 to 0.084, because a 15 A neighbourhood holds thousands of pairs
    # against a few dozen bonds and the bonds are simply outvoted.
    local_distance_loss: bool = False
    # Restrict the local term to ligand rows, as ``bond12``/``bond13`` were.
    # Measured to matter: running it over every atom kept the protein side
    # intact (TM 0.852, lDDT 0.952, both equal to the hand-weighted arm) but
    # took ligand PB-valid from 0.685 to 0.124 and bond MAE from 0.075 to
    # 0.189 A. CrossDocked supplies 9.3 protein atoms per ligand atom, so
    # inside one term the ligand's bonds are simply outvoted -- the same shape
    # as long-range outvoting local, one scale down. The per-source split is
    # therefore not an arbitrary choice: it is what gives the ligand a budget.
    local_distance_ligand_only: bool = False

    pair_distance_loss: bool = False
    # lDDT's own inclusion radius, so the term covers the pairs the metric
    # covers. The only hand-set number left in the geometry objective.
    pair_distance_cutoff: float = 15.0
    # Floor on the reference distance in the denominator. Below any real bond,
    # so it only guards against a degenerate pair, never against real geometry.
    pair_distance_floor: float = 0.3
    # Keep the clash hinge alongside the pairwise term. Off by default: the
    # point of the pairwise term is that it subsumes it.
    keep_clash: bool = False
    # Drop the clash hinge outright. It carries the last hand-set weight in the
    # geometry objective (5.0), so whether it is still needed once 1-2/1-3 are
    # constrained is worth an experiment rather than an assumption.
    drop_clash: bool = False

    distance_map_loss: bool = False
    # 15 A is lDDT's own inclusion radius (Mariani et al., 2013), so the term
    # covers the pairs the metric covers rather than a radius someone picked.
    distance_map_cutoff: float = 15.0

    # How the per-head losses are combined. The hand-set ``recon_weights`` below
    # could not be defended -- ``numH`` sat at 0.1 against ``bond12``'s 5.0, and
    # the resulting 0.981 -> 0.930 drop in per-atom numH accuracy cost more at
    # the molecule level (0.660 -> 0.299 SMILES recovery) than the geometry gain
    # was worth -- so the balance should not be a number someone picked.
    #
    # ``"none"``   use ``recon_weights`` as written (every published run).
    # ``"scale"``  divide each head by a detached running mean of itself, so the
    #              heads contribute equally in RELATIVE terms. MEASURED TO
    #              DIVERGE, for the same reason as "uncertainty" below: the
    #              CONTRIBUTION is pinned near 1, but the WEIGHT is 1/mean, and
    #              a head whose loss approaches zero therefore takes an
    #              unbounded one. Run ``vq_scale`` (job 8369518) reached
    #              aromatic=7.6e6, charge=1.2e6 and element=4.8e4 against
    #              coord=0.012; the chemistry heads hit 0.99999 accuracy and
    #              val/atom_coord sat at 85.8 where the control reached 0.0958.
    #              Equal relative contribution is not the objective, and buying
    #              it costs the objective entirely.
    # ``"constrained"`` treat geometry as the objective and each chemistry head
    #              as a CONSTRAINT at the level the control run reached, with a
    #              Lagrange multiplier raised by dual ascent whenever the
    #              constraint is violated. Unlike the two balancers this encodes
    #              the actual goal -- improve geometry without regressing
    #              chemistry -- which is not a balance: measured with "scale",
    #              equalising the contributions hands ``coord`` (the largest raw
    #              loss, and the one that matters) a weight of 0.017. Bounded by
    #              construction -- the multiplier lives in [0, 1] -- so it cannot
    #              end like "uncertainty" below.
    # ``"uncertainty"`` homoscedastic uncertainty weighting (Kendall et al.,
    #              CVPR 2018). MEASURED TO DIVERGE HERE: minimising
    #              ``exp(-s)L + s/2`` puts the optimum at ``exp(-s) = 1/(2L)``,
    #              so a head whose loss approaches zero takes an unbounded
    #              weight. Run ``vq_uw`` reached charge=7.3e6 and bb_sc=4.0e6
    #              while coord fell to 0.39, and val/atom_coord rose from 1.166
    #              to 1.213. Kept for the record, not for use.
    loss_balancing: str = "none"

    # Decay of the running mean used by ``loss_balancing="scale"``.
    loss_scale_decay: float = 0.99

    # Constraint levels for ``loss_balancing="constrained"``: the per-head
    # TRAINING losses run ``vq_ctrl_p3`` actually reached (job 8354545, 100
    # epochs, CASF/sbdd held out). Measured, not chosen -- which is the point:
    # the constraint says "do not do worse than the control did", and the
    # control is a run that exists.
    constraint_targets: dict[str, float] = field(
        default_factory=lambda: {
            "element": 0.002134,
            "charge": 0.000750,
            "hybrid": 0.004213,
            "aromatic": 0.000741,
            "ring": 0.012842,
            "numH": 0.011291,
            "aa": 0.000126,
            "bb_sc": 0.000177,
        }
    )
    # Dual-ascent step for the multipliers. The violation is divided by the
    # target and clipped to +-1 before it is applied, so one rate serves heads
    # whose targets span two orders of magnitude and a multiplier needs ~1/lr
    # steps to cross its whole range rather than saturating on batch one.
    constraint_lr: float = 0.01

    # Floor (A) for the ligand pair term. The term fires only on pairs the
    # REFERENCE keeps at least this far apart, so raising it cannot punish a
    # bond or a ring's 1-3 contact -- it only forbids the decoder from inventing
    # a close contact that the crystal does not have. At the historical 1.2 A
    # the term is nearly inert: 68.5% of reconstructions still contain a
    # non-bonded pair under 2.0 A.
    clash_floor: float = 1.2

    # --- codebook balance (R2) ------------------------------------------
    #
    # Weight applied to ligand atoms in both the reconstruction loss and the
    # codebook's EMA update. CrossDocked gives 8.3 protein atoms per ligand
    # atom, so at 1.0 the shared book's centroids are set by protein geometry:
    # only 5 of 8192 codes end up ligand-exclusive, and ligand atoms quantized
    # to a shared code are 24% less accurate than those on an exclusive one.
    # 8.3 equalises the two modalities' influence; 1.0 reproduces every
    # published run.
    #
    # This is the LOSS weight. ``ligand_ema_weight`` below splits off the other
    # half, because the two are different mechanisms and only the split can say
    # which one matters: the EMA weight decides how many codes the ligand gets,
    # the loss weight decides how hard the encoder works to use them. Measured
    # on CASP16, the separate-tokenizer ablation -- whose ligand owns a private
    # 4096-code book -- reconstructs ligand chemistry at 0.9804 per atom against
    # joint's 0.9370, so one of the two is costing the joint model real
    # accuracy; which one is an open question this exists to answer.
    ligand_source_weight: float = 1.0

    # Weight applied to ligand atoms in the codebook's EMA update alone. None
    # means "whatever ``ligand_source_weight`` is", which is how the two were
    # coupled before they were separable -- so every existing run and every
    # config that does not set this behaves exactly as it did.
    ligand_ema_weight: float | None = None


@dataclass
class AtomVQVAETrainingConfig:
    """Config for unified all-atom VQ-VAE training (single stream / codebook).

    ``mol_batch_size`` is much smaller than the legacy ligand-only VQ-VAE:
    protein-atom sequences are ~10x longer (a whole pocket), so padding a large
    batch to the pocket length would OOM. Tune against the regenerated
    all-atom cache.
    """

    learning_rate: float = 3e-4
    mol_batch_size: int = 256
    max_epochs: int = 100
    num_workers: int = 16
    precision: str = "bf16-mixed"
    atom: AtomVQVAEConfig = field(default_factory=AtomVQVAEConfig)
    pocket: PocketExtractionConfig = field(default_factory=PocketExtractionConfig)


    # Seed for this run, recorded so a checkpoint remembers what it was trained
    # with. Safe to add: a dataclass field with a default is also a class
    # attribute, so checkpoints pickled before it existed still read
    # ``cfg.seed`` and get this default. A field *without* a default would not
    # have been.
    seed: int = DEFAULT_SEED


# ---------------------------------------------------------------------------
# Autoregressive LM over VQ-VAE tokens (项目(2))
# ---------------------------------------------------------------------------
#
# A dense Qwen3-style decoder trained from scratch on VQ-VAE codebook tokens.
# Sequences are ``<bos><p> protein-pocket tokens </p><l> ligand tokens </l><eos>``
# in a flat vocabulary (see :mod:`prolit.tokenizers.lm_vocab`). The default model
# dims target ~0.3B parameters, sized down from Qwen3-0.6B because the training
# corpus is only ~1B tokens (token/param ≈ 3).

# Number of special tokens; must match ``lm_vocab.NUM_SPECIAL``.
_NUM_SPECIAL_TOKENS = 7


@dataclass
class ProLITCLMConfig:
    """Architecture config for the dense Qwen3-style autoregressive LM (~0.3B)."""

    # Size of the single all-atom code range the flat vocabulary is built on
    # (``vocab_size = specials + atom_codebook_size``). 8192 is the joint
    # tokenizer; the separate-tokenizer ablation passes the COMBINED size of its
    # two sub-codebooks -- also 8192, since it is 4096 per modality -- because
    # both arms stitch their books into one contiguous range before the LM sees
    # them.
    atom_codebook_size: int = 8192

    # Dims chosen to land at ~0.3B parameters with the ~12.3k vocabulary
    # (measured: 302M). Token/param ≈ 3 against the ~1B-token corpus.
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 64
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    # Attention backend: "sdpa" (fast, default) or "eager" (most permissive
    # with custom 4D block-diagonal masks if a transformers version balks).
    attn_implementation: str = "sdpa"

    @property
    def vocab_size(self) -> int:
        return _NUM_SPECIAL_TOKENS + self.atom_codebook_size


@dataclass
class CLMTrainingConfig:
    """Config for from-scratch training of the autoregressive ligand LM."""

    token_dir: Path = Path("data/lm_tokens")
    block_size: int = 512
    # Sequences packed per device per step (a "micro batch" of packed blocks).
    micro_batch_size: int = 64
    gradient_accumulation: int = 1
    # condition-only fine-tuning: mask the ``<p> pocket </p>`` prompt from the
    # loss so only the generated ``<l>`` ligand block is trained. Leave False
    # for pretraining (protein-only / ligand-only), where loss on all tokens
    # teaches the marginals.
    mask_prompt: bool = False
    #: Multiply the loss on the first ``anchor_loss_atoms`` ligand tokens of
    #: each document by this. 1.0 = off (every existing checkpoint).
    anchor_loss_weight: float = 1.0
    #: Weight of an auxiliary head that regresses the ligand centroid from the
    #: ``<l>`` hidden state. 0 = off. Needs ``code_mean_coords``.
    centroid_loss_weight: float = 0.0
    code_mean_coords: str = ""
    #: Width (A) of the geometry-smoothed cross-entropy. 0 = plain CE, which
    #: is what every existing checkpoint was trained with.
    #:
    #: Cross-entropy over a codebook treats every wrong code as equally wrong.
    #: These codes are not symbols -- each one puts an atom somewhere, and the
    #: table in ``code_mean_coords`` says where. Measured, the LM's
    #: teacher-forced argmax lands atoms 2.52 A from the crystal against the
    #: quantizer's own 0.35 A, so nearly all of the deployed pose error is the
    #: LM picking a *geometrically distant* code rather than a near one -- a
    #: distinction the loss it was trained with cannot see. At ``tau`` the
    #: target becomes a Gaussian over the true code's neighbours in that table
    #: instead of a one-hot, so a near miss costs less than a far one. Needs
    #: ``code_mean_coords``.
    code_geometry_tau: float = 0.0
    #: How many neighbours the smoothed target spreads over.
    code_geometry_k: int = 32
    #: What the auxiliary head regresses: "centroid" (the molecule's centre) or
    #: "anchor" (the first ligand atom, which is what the anchor token needs).
    centroid_target: str = "centroid"
    anchor_loss_atoms: int = 3
    #: Fraction of TRAINING documents whose pocket is blanked to an empty
    #: ``<p></p>``, giving the model an unconditional branch. Without one,
    #: classifier-free guidance has nothing valid to extrapolate from.
    pocket_dropout: float = 0.0

    learning_rate: float = 6e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 2000

    max_epochs: int = 2
    num_workers: int = 8
    precision: str = "bf16-mixed"

    model: ProLITCLMConfig = field(default_factory=ProLITCLMConfig)

    # Seed for this run, recorded so a checkpoint remembers what it was trained
    # with. Safe to add: a dataclass field with a default is also a class
    # attribute, so checkpoints pickled before it existed still read
    # ``cfg.seed`` and get this default. A field *without* a default would not
    # have been.
    seed: int = DEFAULT_SEED


@dataclass
class ProLITMLMConfig:
    """Architecture config for the self-implemented ESM-style complex-token MLM.

    A from-scratch bidirectional transformer encoder (rotary attention, pre-LN
    blocks, tied MLM head) -- ESM-*style* but not HuggingFace's ``EsmModel``.
    Same VQ-VAE token vocabulary as the decoder LM, plus one appended ``<mask>``
    token at id ``base_vocab_size`` (so existing codebook offsets are untouched
    and the token caches remain valid). The MLM head predicts over the full
    embedding table; the ``<mask>`` id is only ever an input, never a target.
    """

    # Base vocabulary, mirroring ProLITCLMConfig: one all-atom code range.
    atom_codebook_size: int = 8192

    # ~100M-param ESM3-style encoder: 13 x (hidden 768, 12 heads, SwiGLU 8/3).
    # head_dim = 64. Faithful to Biohub/esm UnifiedTransformerBlock: rotary attn,
    # QK-LayerNorm, SwiGLU FFN (hidden rounded to a 256-multiple), residual
    # scaling by sqrt(n_layers/36), final LayerNorm(bias=False), bias-free.
    hidden_size: int = 768
    num_hidden_layers: int = 13
    num_attention_heads: int = 12
    ffn_expansion_ratio: float = 8 / 3
    qk_layernorm: bool = True
    bias: bool = False
    scale_residue: bool = True
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    # Upper bound for the rotary cos/sin cache; complex docs are <= ~496.
    max_position_embeddings: int = 1024
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def base_vocab_size(self) -> int:
        """Number of real token ids (specials + codebook), before ``<mask>``."""
        return _NUM_SPECIAL_TOKENS + self.atom_codebook_size

    @property
    def mask_token_id(self) -> int:
        """``<mask>`` appended after all real ids (does not shift codebook offsets)."""
        return self.base_vocab_size

    @property
    def vocab_size(self) -> int:
        """Embedding / head size: base vocabulary + the appended ``<mask>``."""
        return self.base_vocab_size + 1


@dataclass
class MLMTrainingConfig:
    """Config for from-scratch training of the bidirectional complex-token MLM."""

    token_dir: Path = Path("data/lm_tokens_finetune_mixed")
    # Max sequence length (complex docs are <= ~496; block_size caps/truncates).
    block_size: int = 512
    micro_batch_size: int = 64
    gradient_accumulation: int = 1

    # BERT-style dynamic masking. Of the ``mask_prob`` selected positions:
    # ``mask_replace_prob`` -> <mask>, ``mask_random_prob`` -> random codebook
    # token, remainder -> unchanged. Only codebook tokens are ever masked
    # (specials <p>/<l>/<bos>/... are excluded so structure markers stay intact).
    mask_prob: float = 0.15
    #: Upper end of a per-example uniform mask rate. 0 keeps the fixed
    #: ``mask_prob`` every existing checkpoint was trained with. Set it to 1.0
    #: to span the whole schedule an iterative (MaskGIT-style) decoder walks
    #: through, which starts from a fully masked ligand.
    mask_prob_max: float = 0.0
    mask_replace_prob: float = 0.8
    mask_random_prob: float = 0.1
    #: Path to an (n_codes, K) int16 table of each code's nearest neighbours in
    #: codebook space. When set, the ``mask_random_prob`` share is drawn from
    #: those neighbours instead of uniformly over the vocabulary, so the
    #: corruption is a near miss rather than an obvious intruder. Empty = the
    #: uniform draw every existing checkpoint was trained with.
    code_neighbours: str = ""
    # When True, only ligand (``<l>..</l>``) tokens are masked -> the model learns
    # P(ligand | pocket) bidirectionally (a condition-only / rescoring-tuned MLM).
    ligand_only_masking: bool = False

    learning_rate: float = 4e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    grad_clip: float = 1.0
    warmup_steps: int = 2000

    max_epochs: int = 10
    num_workers: int = 8
    precision: str = "bf16-mixed"

    model: ProLITMLMConfig = field(default_factory=ProLITMLMConfig)

    # Seed for this run, recorded so a checkpoint remembers what it was trained
    # with. Safe to add: a dataclass field with a default is also a class
    # attribute, so checkpoints pickled before it existed still read
    # ``cfg.seed`` and get this default. A field *without* a default would not
    # have been.
    seed: int = DEFAULT_SEED


@dataclass
class RescoreTrainingConfig:
    """Config for fine-tuning a pose-scoring head on the pretrained MLM encoder.

    The encoder (:class:`ProLITMLMConfig`) is warm-started from a pretrained MLM
    checkpoint; a small MLP head over the mean-pooled ligand-token representations
    regresses the pose RMSD (lower = more native-like). Trained on RMSD-labelled
    decoys (:mod:`pipelines.corpora.tokenize_decoys`).
    """

    token_dir: Path = Path("data/lm_tokens_decoys")
    block_size: int = 512
    micro_batch_size: int = 32
    gradient_accumulation: int = 1

    head_dropout: float = 0.1
    # Target RMSDs are clipped here (a 12 A decoy is no worse than an 8 A one for
    # ranking) and losses beyond ~this add noise.
    rmsd_cap: float = 8.0

    # Pairwise ranking loss (docking power is a ranking task, not regression).
    # When > 0, batches are grouped by complex (native pose has RMSD 0.0 marks
    # each group boundary) and a margin loss pushes pred(lower-RMSD) below
    # pred(higher-RMSD) within each complex, added to the regression loss.
    ranking_loss_weight: float = 0.0
    ranking_margin: float = 0.5

    # ListNet-style listwise loss over the poses of one complex: softmax(-pred)
    # is matched to softmax(-rmsd / tau). Unlike the pairwise margin loss it
    # spends its gradient on the near-native end (which pose wins) instead of
    # weighting every pose pair equally, and the model stays a per-pose scorer.
    # Cap the number of TRAIN docs (0 = all). Prefix of the corpus, so it takes
    # whole complexes; used to compare corpora at matched size.
    max_docs: int = 0

    # Drop training poses whose label exceeds this (0 = keep all). Trains a
    # near-native specialist for the top-1 tie-break.
    max_label: float = 0.0

    # Drop the RMSD-0 crystal pose from training. CASF is scored decoys-only, so
    # a head that has seen the native learns a pose the benchmark never asks it
    # to rank.
    #
    # Declared here with its default rather than only set by
    # ``scoring_head.py`` when its flag is passed: the dataset reads the
    # attribute unconditionally, so without this every run that did NOT pass
    # --drop-native-pose died in the datamodule's setup (128 of the 130
    # scoring_head commands in jobs/). A dataclass default is also what makes
    # a config pickled into an older checkpoint still resolve the attribute,
    # so adding it does not invalidate existing heads.
    drop_native_pose: bool = False

    # Weight of the per-ligand-atom displacement auxiliary loss (needs a corpus
    # with .disp/.dlen sidecars from tokenize_decoys).
    atom_aux_weight: float = 0.0

    listwise_loss_weight: float = 0.0
    listwise_label_tau: float = 0.4
    listwise_pred_tau: float = 0.4
    complexes_per_batch: int = 8
    # Cap on docs drawn per group in one batch. Needed for the affinity corpus,
    # where a group is a protein and sizes range from 1 to ~700 ligands.
    # 0 = take the whole group (pose corpus: ~20 poses/complex).
    max_per_group: int = 0

    # Ligand-token pooling for the head. "mean" averages (a single bad-contact
    # atom is washed out); "meanmax" concatenates mean + max so the worst atom
    # -- the strongest wrong-pose signal -- survives to the head.
    pooling: str = "mean"

    # Freeze the pretrained encoder and train only the pooling + head. Cuts the
    # trainable capacity from 99M to ~1M so a ranking loss can't memorize the
    # small affinity corpus; the head re-weights fixed features instead.
    freeze_encoder: bool = False

    # Regress ligand efficiency (label / heavy-atom count) instead of the raw
    # label. For the affinity head this decorrelates the target from molecular
    # size; eval multiplies the prediction back by size to recover pK.
    label_divide_by_size: bool = False

    # Number of interaction channels for the "pairsum" pooling (sum of learned
    # ligand-pocket pair energies). Each channel adds one feature to the head.
    pair_heads: int = 16

    # Trainable transformer layers inserted over the token states before pooling
    # (0 = none). Gives the head capacity to re-model the interface from the
    # tokens without touching the tokenizer.
    head_interaction_layers: int = 0

    # Masked-LM auxiliary loss during affinity fine-tuning: keeps the encoder's
    # structure knowledge intact (regularizer) while a ranking loss adapts it to
    # affinity, so the ranking objective can't collapse/memorize the small corpus
    # the way it did head-only. 0 = off.
    mlm_aux_weight: float = 0.0
    mlm_aux_mask_prob: float = 0.15

    learning_rate: float = 1e-4  # low: the encoder is pretrained
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    grad_clip: float = 1.0
    warmup_steps: int = 500

    max_epochs: int = 10
    num_workers: int = 8
    precision: str = "bf16-mixed"

    model: ProLITMLMConfig = field(default_factory=ProLITMLMConfig)

    # Seed for this run, recorded so a checkpoint remembers what it was trained
    # with. Safe to add: a dataclass field with a default is also a class
    # attribute, so checkpoints pickled before it existed still read
    # ``cfg.seed`` and get this default. A field *without* a default would not
    # have been.
    seed: int = DEFAULT_SEED


@dataclass
class PoseRefinerConfig:
    """Architecture config for the E(3)-equivariant pose refiner (e3nn).

    A small equivariant graph denoiser that refines *ligand* heavy-atom
    coordinates conditioned on the *frozen* pocket atoms. Nodes carry invariant
    (``0e``) chemistry scalars; edges carry spherical harmonics of their
    direction (degree ``<= l_max``) gated by a radial MLP of the distance. The
    network emits one ``1o`` vector per ligand atom = a coordinate displacement,
    so ``refine(R x, R pkt) = R refine(x, pkt)`` for any rotation/reflection.

    The refiner is trained as a flow-matching bridge from the VQ-VAE
    reconstruction of a native ligand (``x0``, the exact deployment corruption)
    to the crystal pose (``x1``); see :class:`PoseRefineTrainingConfig`.
    """

    # Invariant scalar width carried on each node (0e channels).
    hidden_dim: int = 128
    # Number of equivariant convolution layers.
    n_layers: int = 5
    # Max spherical-harmonic degree for edge features and hidden irreps.
    l_max: int = 2
    # Radial basis (Bessel) count + MLP width for the distance embedding.
    num_radial: int = 16
    radial_hidden: int = 64

    # Graph construction. Ligand-ligand edges are full O(N^2) when
    # ``ligand_knn == 0`` (N_heavy <= ~50), else k-NN. Ligand-pocket edges use a
    # radius cutoff; pocket-pocket edges are never built (pocket geometry is
    # fixed and only conditions the ligand).
    ligand_knn: int = 0
    pocket_cutoff: float = 8.0  # angstrom
    max_pocket_neighbors: int = 32  # per-ligand-atom cap on pocket edges

    # Inference steps from x0 -> refined pose. 1 = single-shot x1-prediction
    # (robust; one forward pass directly estimates the clean pose). >1 uses the
    # multi-step velocity ODE, which compounds per-step error and needs a
    # well-trained field -- kept only for ablation.
    n_flow_steps: int = 1
    # Optional Brownian-bridge noise on the interpolant (stochastic interpolant).
    bridge_sigma: float = 0.0

    # Physical auxiliary losses (mirror ``solve_ligand_coords`` + ``infer_bonds``).
    d_floor: float = 1.1  # angstrom, hard no-overlap floor
    # Minimum separation (A) enforced only between pairs the CRYSTAL keeps at
    # least that far apart, so the floor may exceed a bond length -- a bond, or
    # a ring's 1-3 contact, excludes itself by its own reference distance. This
    # is the term that can actually close invented contacts; ``d_floor`` alone
    # has to stay under a bond length and barely fires. ``None`` keeps the
    # historical behaviour.
    nonbond_floor: float | None = None
    lambda_clash: float = 1.0  # intra-ligand steric
    lambda_pkt: float = 1.0  # ligand-pocket steric
    lambda_bond: float = 1.0  # bonded-distance anchor (anti-collapse / topology)
    lambda_angle: float = 0.0  # bond-angle -> PoseBusters bond-angle validity
    # Steps over which the steric/bond weights ramp in from 0 (learn the
    # reconstruction manifold first, then enforce sterics -> no early collapse).
    lambda_ramp_steps: int = 2000


@dataclass
class BondHeadTrainingConfig:
    """Config for the bond head (:mod:`prolit.model.bond_head`).

    The head replaces distance-based bond perception on the decoder's output,
    where perception recovers 31% of the true graph. Its training data is a
    pose-refine corpus: the LM's own decoded coordinates beside the crystal
    molecule's true bonds.

    Stored in the checkpoint as a plain dict rather than as a pickled instance,
    so renaming or moving this class does not orphan the weights the way it
    would for the Lightning modules.
    """

    data_dir: str = ""
    max_epochs: int = 40
    #: Molecules per optimiser step. A molecule contributes all of its atom
    #: pairs, so this is not the number of training examples.
    batch_molecules: int = 64
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    embedding_dim: int = 48
    hidden_dim: int = 256
    seed: int = DEFAULT_SEED


@dataclass
class PoseRefineTrainingConfig:
    """Config for training the e3nn pose refiner (flow-matching, x1-prediction).

    Source ``x0`` = VQ-VAE round-trip of a real ligand (+ graded corruption),
    target ``x1`` = crystal native pose, both in the pocket canonical frame;
    interpolate ``x_t = (1-t) x0 + t x1`` and regress the clean pose ``x1`` with
    physical auxiliary losses. Data is produced offline by
    :mod:`pipelines.corpora.tokenize_pose_refine` into concatenated memmaps.
    """

    data_dir: Path = Path("data/pose_refine")
    micro_batch_size: int = 32  # graphs (complexes) per batch
    gradient_accumulation: int = 1

    # Online per-atom Gaussian jitter added to x0 at load (train split only).
    # Injects intramolecular distortion (bad bond lengths/angles/clashes) that
    # the refiner must repair -- the VQ round-trip alone is nearly clash-free
    # intramolecularly, so without this the net never learns to fix the internal
    # geometry that PoseBusters checks. Edges are rebuilt from the jittered pose.
    online_jitter_sigma: float = 0.0

    # Online RIGID-BODY corruption of x0 (train split only): random translation
    # (angstrom sigma) and rotation about the ligand centroid (degree sigma).
    # LM-sampled poses are predominantly MIS-PLACED in the pocket -- the driver
    # of the bad raw Vina score -- whereas the VQ round-trip + jitter only model
    # local distortion. Teaching the refiner to slide/tilt the ligand back is
    # what closes the train/inference gap on placement.
    online_rigid_trans: float = 0.0
    online_rigid_rot_deg: float = 0.0
    # Fraction of training samples that get the rigid corruption. Applying it to
    # EVERY sample makes the net over-correct (it never sees an already-correct
    # pose it should leave alone -> val rmsd_gain went NEGATIVE), so keep a
    # sizable share of clean-placement examples.
    online_rigid_prob: float = 0.5
    #: Push the pose toward the pocket's centre of mass by |N(0, this)| A.
    #: The isotropic ``online_rigid_trans`` cannot express this: generated
    #: poses are not randomly offset, they are uniformly ~0.36 A too deep
    #: (median surface gap -0.467 A against reference ligands' -0.104), and
    #: Vina charges for it through repulsion (7.50 vs FLOWR's 1.64) while every
    #: attractive term is already better than FLOWR's.
    online_press_sigma: float = 0.0
    #: Std (radians) of the random torsion applied to each rotatable bond when
    #: corrupting a pose. A torsion-output refiner whose training corruption has
    #: no torsional component learns to emit zero angles -- the head has nothing
    #: to undo -- which collapses it to a rigid-only refiner, and rigid alone is
    #: measured to stop at 16.5% clash against FLOWR's 9.7%.
    online_torsion_sigma: float = 0.0
    #: Weight on DIRECT supervision of the torsion angles (the corruption angle
    #: is known, so it need not be inferred through a coordinate loss). 0 keeps
    #: the coordinate-only objective.
    torsion_angle_weight: float = 0.0

    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    grad_clip: float = 1.0
    warmup_steps: int = 2000

    max_epochs: int = 40
    num_workers: int = 8
    precision: str = "32-true"  # e3nn tensor products are unstable in bf16

    model: PoseRefinerConfig = field(default_factory=PoseRefinerConfig)

    # Seed for this run, recorded so a checkpoint remembers what it was trained
    # with. Safe to add: a dataclass field with a default is also a class
    # attribute, so checkpoints pickled before it existed still read
    # ``cfg.seed`` and get this default. A field *without* a default would not
    # have been.
    seed: int = DEFAULT_SEED

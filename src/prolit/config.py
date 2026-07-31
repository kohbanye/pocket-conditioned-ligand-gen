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
    mask_replace_prob: float = 0.8
    mask_random_prob: float = 0.1
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
    lambda_clash: float = 1.0  # intra-ligand steric
    lambda_pkt: float = 1.0  # ligand-pocket steric
    lambda_bond: float = 1.0  # bonded-distance anchor (anti-collapse / topology)
    lambda_angle: float = 0.0  # bond-angle -> PoseBusters bond-angle validity
    # Steps over which the steric/bond weights ramp in from 0 (learn the
    # reconstruction manifold first, then enforce sterics -> no early collapse).
    lambda_ramp_steps: int = 2000


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


# --- Names these classes were pickled under -------------------------------
#
# Lightning stores a run's config *instance* in ``hyper_parameters``, and pickle
# records a class by (module path, class name). Every checkpoint trained before
# these classes took the paper's names therefore asks ``prolit.config`` for the
# old name, and unpickling calls ``getattr`` on this module -- so an alias is
# enough to keep them loading, and they come back as the new class.
#
# ``prolit/_legacy_import_path.py`` does the same job one level up, for the
# module path. Deleting either one silently orphans trained weights: the failure
# is an ImportError at load time, long after the run that produced them is gone.
#
# ``tests/test_legacy_checkpoint_path.py`` pins this.
LigandLMConfig = ProLITCLMConfig
LMTrainingConfig = CLMTrainingConfig
ComplexMLMConfig = ProLITMLMConfig

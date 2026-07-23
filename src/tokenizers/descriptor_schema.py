"""Descriptor schema for the spherical multi-head VQ-VAE.

Defines the vocabularies and field layouts shared by:
- ``LigandDescriptor`` / ``BackboneSphericalDescriptor`` (encode side)
- ``TransformerVQVAE`` (encoder embeddings + decoder heads)
- ``ComplexDescriptorDataModule`` (Welford normalization, only on continuous slots)

A descriptor row is a single concatenated float32 vector. Continuous slots
hold real values (spherical coords); categorical slots hold integer indices
cast to float (the network casts back to long before embedding lookup).

The layout is documented as ``(start, length)`` tuples in two helpers per
descriptor: ``LIG_LAYOUT`` / ``PROT_LAYOUT``. Tests assert these stay in
sync with the descriptors and the VQ-VAE.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Categorical vocabularies
# ---------------------------------------------------------------------------

# Element vocabulary covers the heavy atoms typically present in CrossDocked2020
# ligands. ``OTHER`` (idx 11) is the catch-all bucket; padding is implicit at
# the sequence level via the attention mask, so we do not reserve a pad index.
LIGAND_ELEMENT_VOCAB: tuple[str, ...] = (
    "C",
    "N",
    "O",
    "S",
    "F",
    "Cl",
    "Br",
    "I",
    "P",
    "B",
    "Si",
    "OTHER",
)
LIGAND_ELEMENT_TO_IDX: dict[str, int] = {
    e: i for i, e in enumerate(LIGAND_ELEMENT_VOCAB)
}
LIGAND_OTHER_IDX = LIGAND_ELEMENT_TO_IDX["OTHER"]

# Formal charge: clamp anything outside [-2, +2] to the boundary.
LIGAND_CHARGE_VOCAB: tuple[int, ...] = (-2, -1, 0, 1, 2)
LIGAND_CHARGE_TO_IDX: dict[int, int] = {c: i for i, c in enumerate(LIGAND_CHARGE_VOCAB)}

# Hybridization: SP, SP2, SP3, aromatic, other (covers SP3D / S / UNSPECIFIED).
# Aromatic is its own bucket because RDKit reports aromatic atoms as SP2 with
# an aromatic flag, and the codebook benefits from seeing them as a distinct
# state.
LIGAND_HYBRID_VOCAB: tuple[str, ...] = ("SP", "SP2", "SP3", "AROM", "OTHER")
LIGAND_HYBRID_OTHER_IDX = 4

# Smallest ring containing the atom: 3 / 4 / 5 / 6+ / not-in-ring.
LIGAND_RING_VOCAB: tuple[str, ...] = ("R3", "R4", "R5", "R6+", "NONE")
LIGAND_RING_NONE_IDX = 4

# Total H count (implicit + explicit), clamped to [0, 4].
LIGAND_NUMH_VOCAB: tuple[int, ...] = (0, 1, 2, 3, 4)

# 20 standard amino acids + X. Order matches ``ProteinSequenceTokenizer``.
PROTEIN_AA_VOCAB: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWYX")
PROTEIN_AA_TO_IDX: dict[str, int] = {a: i for i, a in enumerate(PROTEIN_AA_VOCAB)}
PROTEIN_AA_X_IDX = PROTEIN_AA_TO_IDX["X"]

# ---------------------------------------------------------------------------
# Unified all-atom descriptor vocabularies (protein + ligand heavy atoms share
# one descriptor and one codebook). The ``source`` flag tells the encoder which
# domain an atom belongs to; ``bb_sc`` marks protein backbone vs side chain.
# ---------------------------------------------------------------------------

# Atom source: protein pocket atom (0) vs ligand atom (1). Carried as an input
# feature only (no reconstruction head): it is always known at encode time and
# conditions the shared codebook, while staying implicit in the code itself.
SOURCE_VOCAB: tuple[str, ...] = ("protein", "ligand")
SOURCE_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(SOURCE_VOCAB)}
SOURCE_PROTEIN_IDX = SOURCE_TO_IDX["protein"]
SOURCE_LIGAND_IDX = SOURCE_TO_IDX["ligand"]

# Backbone / side-chain / not-applicable. Protein backbone atoms are
# ``N, CA, C, O, OXT``; everything else in a residue is a side-chain atom.
# Ligand atoms have no notion of backbone, so they take the ``NA`` bucket.
BB_SC_VOCAB: tuple[str, ...] = ("backbone", "sidechain", "NA")
BB_SC_BACKBONE_IDX = 0
BB_SC_SIDECHAIN_IDX = 1
BB_SC_NA_IDX = 2
PROTEIN_BACKBONE_ATOM_NAMES: frozenset[str] = frozenset({"N", "CA", "C", "O", "OXT"})


# ---------------------------------------------------------------------------
# Field layouts
# ---------------------------------------------------------------------------

# K nearest neighbours stored as encoder hint features (Mol-StrucTok style).
# Larger K does not help the codebook find better codes once K covers the
# atom's first chemical shell; 4 captures the standard sp3 valence.
K_NEIGHBORS = 4


@dataclass(frozen=True)
class FieldSpec:
    """One contiguous slice of the descriptor vector."""

    start: int
    length: int
    name: str
    kind: str  # "continuous" | "categorical"
    vocab_size: int = 0  # 0 for continuous

    @property
    def end(self) -> int:
        return self.start + self.length


def _build_layout(specs: list[tuple[str, str, int, int]]) -> list[FieldSpec]:
    """Compute absolute offsets from a list of ``(name, kind, length, vocab)``."""
    layout: list[FieldSpec] = []
    cursor = 0
    for name, kind, length, vocab in specs:
        layout.append(
            FieldSpec(
                start=cursor,
                length=length,
                name=name,
                kind=kind,
                vocab_size=vocab,
            )
        )
        cursor += length
    return layout


# Ligand atom descriptor: 30-D total
#   - 4 continuous spherical (r, theta, sin phi, cos phi) from pocket centroid
#   - 6 categorical singletons (element, charge, hybrid, aromatic, ring, numH)
#   - 16 continuous KNN spherical offsets (K=4 atoms x 4D each)
#   - 4 categorical KNN element indices (one per neighbour)
LIGAND_LAYOUT: list[FieldSpec] = _build_layout(
    [
        ("coord", "continuous", 4, 0),
        ("element", "categorical", 1, len(LIGAND_ELEMENT_VOCAB)),
        ("charge", "categorical", 1, len(LIGAND_CHARGE_VOCAB)),
        ("hybrid", "categorical", 1, len(LIGAND_HYBRID_VOCAB)),
        ("aromatic", "categorical", 1, 2),
        ("ring", "categorical", 1, len(LIGAND_RING_VOCAB)),
        ("numH", "categorical", 1, len(LIGAND_NUMH_VOCAB)),
        ("knn_offsets", "continuous", 4 * K_NEIGHBORS, 0),
        ("knn_elements", "categorical", K_NEIGHBORS, len(LIGAND_ELEMENT_VOCAB)),
    ]
)
LIGAND_DESCRIPTOR_DIM: int = LIGAND_LAYOUT[-1].end  # 30

# Protein residue descriptor: 65-D total
#   - 12 continuous: 3 atoms (N, CA, C) x 4 spherical from pocket centroid
#   - 1 categorical: amino acid identity
#   - 48 continuous KNN residue spherical offsets (K=4 residues x 12D each)
#   - 4 categorical KNN residue AA indices
PROTEIN_LAYOUT: list[FieldSpec] = _build_layout(
    [
        ("coord", "continuous", 12, 0),
        ("aa", "categorical", 1, len(PROTEIN_AA_VOCAB)),
        ("knn_offsets", "continuous", 12 * K_NEIGHBORS, 0),
        ("knn_aa", "categorical", K_NEIGHBORS, len(PROTEIN_AA_VOCAB)),
    ]
)
PROTEIN_DESCRIPTOR_DIM: int = PROTEIN_LAYOUT[-1].end  # 65

# Unified all-atom descriptor: 33-D total. One row per heavy atom, used for
# BOTH protein pocket atoms and ligand atoms (one VQ-VAE, one codebook).
#   - 4 continuous spherical (r, θ, sin φ, cos φ) from the pocket centroid
#   - 1 categorical source flag (protein / ligand) — input only, no recon head
#   - 6 categorical chemistry slots shared by both domains: element, charge,
#     hybrid, aromatic, ring, numH (Full ligand-parity: protein atoms get these
#     from an RDKit parse of the receptor too)
#   - 2 categorical protein-context slots: residue type (aa) and backbone/
#     side-chain flag (bb_sc). Ligand atoms take the ``X`` / ``NA`` buckets and
#     their loss on these heads is masked out.
#   - 16 continuous KNN spherical offsets (K=4 same-source atoms x 4D)
#   - 4 categorical KNN element indices
ATOM_LAYOUT: list[FieldSpec] = _build_layout(
    [
        ("coord", "continuous", 4, 0),
        ("source", "categorical", 1, len(SOURCE_VOCAB)),
        ("element", "categorical", 1, len(LIGAND_ELEMENT_VOCAB)),
        ("charge", "categorical", 1, len(LIGAND_CHARGE_VOCAB)),
        ("hybrid", "categorical", 1, len(LIGAND_HYBRID_VOCAB)),
        ("aromatic", "categorical", 1, 2),
        ("ring", "categorical", 1, len(LIGAND_RING_VOCAB)),
        ("numH", "categorical", 1, len(LIGAND_NUMH_VOCAB)),
        ("aa", "categorical", 1, len(PROTEIN_AA_VOCAB)),
        ("bb_sc", "categorical", 1, len(BB_SC_VOCAB)),
        ("knn_offsets", "continuous", 4 * K_NEIGHBORS, 0),
        ("knn_elements", "categorical", K_NEIGHBORS, len(LIGAND_ELEMENT_VOCAB)),
    ]
)
ATOM_DESCRIPTOR_DIM: int = ATOM_LAYOUT[-1].end  # 33


def fields_by_name(layout: list[FieldSpec]) -> dict[str, FieldSpec]:
    return {f.name: f for f in layout}


def continuous_mask(layout: list[FieldSpec]) -> list[bool]:
    """Return a per-dim mask: True for continuous slots, False for categorical.

    Used by the Welford normalization pass to skip categorical columns
    (their stats must be left at mean=0, std=1 so values pass through
    unchanged).
    """
    mask: list[bool] = []
    for spec in layout:
        mask.extend([spec.kind == "continuous"] * spec.length)
    return mask


# Heads the decoder must produce, in fixed order (used by VQ-VAE multi-head
# decoder + reconstruction loss). Continuous heads are predicted as raw
# regression outputs; categorical heads as logits over their vocab.
LIGAND_RECON_HEADS: list[tuple[str, str, int]] = [
    ("coord", "continuous", 4),  # spherical: r, theta, sin phi, cos phi
    ("element", "categorical", len(LIGAND_ELEMENT_VOCAB)),
    ("charge", "categorical", len(LIGAND_CHARGE_VOCAB)),
    ("hybrid", "categorical", len(LIGAND_HYBRID_VOCAB)),
    ("aromatic", "categorical", 2),
    ("ring", "categorical", len(LIGAND_RING_VOCAB)),
    ("numH", "categorical", len(LIGAND_NUMH_VOCAB)),
]

PROTEIN_RECON_HEADS: list[tuple[str, str, int]] = [
    ("coord", "continuous", 12),  # 3 atoms x 4 spherical dims
    ("aa", "categorical", len(PROTEIN_AA_VOCAB)),
]

# Unified all-atom decoder heads. ``source`` is intentionally absent (input
# only). ``coord`` + the six chemistry heads are trained on every atom; the
# protein-context heads (``aa``, ``bb_sc``) are only meaningful for protein
# atoms, so their loss is masked to ``source == protein`` rows in the VQ-VAE.
ATOM_RECON_HEADS: list[tuple[str, str, int]] = [
    ("coord", "continuous", 4),  # spherical: r, theta, sin phi, cos phi
    ("element", "categorical", len(LIGAND_ELEMENT_VOCAB)),
    ("charge", "categorical", len(LIGAND_CHARGE_VOCAB)),
    ("hybrid", "categorical", len(LIGAND_HYBRID_VOCAB)),
    ("aromatic", "categorical", 2),
    ("ring", "categorical", len(LIGAND_RING_VOCAB)),
    ("numH", "categorical", len(LIGAND_NUMH_VOCAB)),
    ("aa", "categorical", len(PROTEIN_AA_VOCAB)),
    ("bb_sc", "categorical", len(BB_SC_VOCAB)),
]

# Heads whose reconstruction loss is restricted to protein atoms (ligand rows
# carry placeholder ``X`` / ``NA`` targets that must not be learned).
ATOM_PROTEIN_ONLY_HEADS: frozenset[str] = frozenset({"aa", "bb_sc"})

"""The stapled baseline's stream must round-trip, or its LM is measuring noise.

Three ranges share one id space here (ESM3 pocket codes, pose digits, ConfSeq
tokens) and the pose rides inside the ligand block. Both are places where an
off-by-one is invisible: the model still trains, the loss still falls, and the
ligand comes out somewhere plausible but wrong. These pin the layout.
"""

from __future__ import annotations

import numpy as np

from prolit.tokenizers.lm_vocab import BOS_ID, EOS_ID, L_OPEN_ID, P_OPEN_ID
from prolit.tokenizers.pose_budget import SemanticPoseCodec
from prolit.tokenizers.stapled import (
    ESM3_CODEBOOK_SIZE,
    ConfSeqVocab,
    StapledVocab,
    place,
    pocket_box,
    pose_of,
)
from prolit.tokenizers.stapled_encoder import build_vocab, merge_vocabs

VOCAB = ConfSeqVocab(tuple(f"t{i}" for i in range(391)))


def _vocab() -> StapledVocab:
    return StapledVocab(confseq=VOCAB)


def test_ranges_are_disjoint_and_cover_the_vocabulary() -> None:
    v = _vocab()
    assert v.n_pose_tokens == 4
    assert 51.0 < v.pose_bits < 52.0
    assert v.esm3_offset == 7
    assert v.pose_offset == 7 + ESM3_CODEBOOK_SIZE
    assert v.confseq_offset == v.pose_offset + 8192
    assert v.vocab_size == v.confseq_offset + 391


def test_sequence_round_trip() -> None:
    v = _vocab()
    esm3 = [0, 4095, 17, 900]
    confseq = [0, 390, 12]
    pose = (7, 8000, 1, 8191)
    seq = v.build_sequence(esm3, confseq, pose)
    assert seq[0] == BOS_ID
    assert seq[-1] == EOS_ID
    assert seq[1] == P_OPEN_ID
    assert L_OPEN_ID in seq
    got_esm3, got_pose, got_confseq = v.split_sequence(seq)
    assert got_esm3 == esm3
    assert got_confseq == confseq
    assert got_pose == pose


def test_pose_digits_lead_the_ligand_block() -> None:
    """Placement is predicted before the molecule, not after it.

    An autoregressive model that emitted the conformer first and the placement
    last would be choosing where to put a molecule it had already committed to,
    which is not the factorization this baseline is meant to represent.
    """
    v = _vocab()
    seq = v.build_sequence([1, 2], [3, 4, 5], pose=(7, 1, 2, 3))
    start = seq.index(L_OPEN_ID) + 1
    digits = seq[start : start + v.n_pose_tokens]
    assert all(v.pose_offset <= d < v.confseq_offset for d in digits)
    assert v.confseq_offset <= seq[start + v.n_pose_tokens] < v.vocab_size


def test_ligand_only_document_carries_no_pose() -> None:
    """A GEOM conformer has no pocket, so there is nothing to place it in."""
    v = _vocab()
    seq = v.build_sequence([], [1, 2, 3], pose=None)
    esm3, pose, confseq = v.split_sequence(seq)
    assert esm3 == []
    assert pose is None
    assert confseq == [1, 2, 3]


def test_truncated_pose_is_none_not_wrong() -> None:
    """A short generation must not decode as a confident placement."""
    v = _vocab()
    seq = v.build_sequence([1], [2, 3], pose=(9, 9, 9, 9))
    start = seq.index(L_OPEN_ID) + 1
    truncated = seq[:start] + seq[start + 1 :]  # drop one pose digit
    _, pose, _ = v.split_sequence(truncated)
    assert pose is None


def test_out_of_vocabulary_confseq_token_is_reported() -> None:
    """Silently dropping a token would change the molecule."""
    assert VOCAB.ids(["t0", "t1"]) == [0, 1]
    assert VOCAB.ids(["t0", "<never-seen>"]) is None


def test_vocab_order_is_deterministic() -> None:
    """Ids are baked into every stream and into the embedding table."""
    counts = {"b": 3, "a": 3, "c": 10}
    assert ConfSeqVocab.from_counts(counts).tokens == ("c", "a", "b")


def test_pose_tokens_are_hierarchical() -> None:
    """Two nearby poses agree on the coarse tokens and differ only in the fine.

    This is the whole reason the LM arm does not reuse the reconstruction
    sweep's packed integer: there the digit boundaries cut across the
    translation raster, so a 0.3 A move can change every token. A model reading
    that has to memorise a hash. Here it reads a refinement.
    """
    rng = np.random.default_rng(11)
    backbone = rng.uniform(-14.0, 14.0, size=(200, 3))
    own = rng.uniform(-3.0, 3.0, size=(18, 3))
    origin, size = pocket_box(backbone)
    codec = SemanticPoseCodec()
    base = own - own.mean(axis=0) + (origin + size * 0.5)

    a = pose_of(base, own, backbone, codec)
    # A nudge well inside one coarse cell (cell = size/20, typically ~1.5 A).
    nudged = base + np.array([size / 200.0, 0.0, 0.0])
    b = pose_of(nudged, own, backbone, codec)
    assert a[0] == b[0], "a sub-cell move changed the coarse translation token"
    assert a[2] == b[2], "a pure translation changed the coarse rotation token"


def test_placement_error_is_below_what_the_tasks_resolve() -> None:
    """The channel must not be the thing the downstream experiment measures.

    Docking power is decided in the 0-1 A band. A quantizer whose own error
    lands in that band would make every language-model result a statement about
    this code rather than about the representation, so it is pinned here.
    """
    rng = np.random.default_rng(5)
    codec = SemanticPoseCodec()
    origin, size = np.array([-15.0, -15.0, -15.0]), 30.0
    worst = 0.0
    for _ in range(60):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        rot = np.array([
            [1 - 2 * (q[2] ** 2 + q[3] ** 2),
             2 * (q[1] * q[2] - q[0] * q[3]),
             2 * (q[1] * q[3] + q[0] * q[2])],
            [2 * (q[1] * q[2] + q[0] * q[3]),
             1 - 2 * (q[1] ** 2 + q[3] ** 2),
             2 * (q[2] * q[3] - q[0] * q[1])],
            [2 * (q[1] * q[3] - q[0] * q[2]),
             2 * (q[2] * q[3] + q[0] * q[1]),
             1 - 2 * (q[1] ** 2 + q[2] ** 2)],
        ])
        centroid = origin + rng.random(3) * size
        got_c, got_r = codec.decode(
            codec.encode(centroid, rot, origin, size), origin, size
        )
        # A drug-sized ligand, 3 A of gyration: rotation error shows up scaled.
        angle = 2 * np.arccos(min(abs(float(np.trace(got_r @ rot.T) - 1) / 2), 1.0))
        worst = max(worst, np.linalg.norm(got_c - centroid) + 3.0 * abs(angle))
    assert worst < 0.25, f"placement error {worst:.3f} A reaches into the 0-1 A band"


def test_placement_recovers_the_pose() -> None:
    """Encode a real transform, decode it, and land back on the ligand."""
    rng = np.random.default_rng(7)
    backbone = rng.uniform(-12.0, 12.0, size=(120, 3))
    own = rng.uniform(-4.0, 4.0, size=(20, 3))
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    origin, size = pocket_box(backbone)
    ligand = (own - own.mean(axis=0)) @ rot.T + (origin + size * 0.5)

    code = pose_of(ligand, own, backbone)
    placed = place(own, code, backbone)
    assert np.abs(placed - ligand).max() < 0.5

    # And with nothing transmitted it lands in the middle of the box, which is
    # the honest zero-budget behaviour rather than an accidental good guess.
    dropped = place(own, None, backbone)
    assert np.allclose(dropped.mean(axis=0), origin + size / 2.0)


def test_merge_vocabs_unions_sources(tmp_path) -> None:  # noqa: ANN001
    """One alphabet over several corpora, and the ids must not depend on which
    pass ran longer.

    Every corpus in a run shares one ConfSeq vocabulary, because the ids are
    baked into each stream and into the embedding table. Ordering the merged
    symbols by frequency would let a longer GEOM pass outrank BioLiP's
    cofactors and silently renumber everything.
    """
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    ConfSeqVocab(("C", "N", "<5>")).save(a)
    ConfSeqVocab(("C", "O", "[nH]")).save(b)

    merged = merge_vocabs([a, b])
    tokens = set(merged.confseq.tokens)
    assert {"C", "N", "O", "[nH]"} <= tokens
    # The geometric half is enumerated, not observed: a decoy that lands on an
    # angle no source happened to take must still be in vocabulary.
    assert {f"<{d}>" for d in (-180, -37, 0, 91, 180)} <= tokens
    assert merged.confseq.size == len(tokens)
    # Deterministic: the same inputs in the other order give the same ids.
    assert merge_vocabs([b, a]).confseq.tokens == merged.confseq.tokens


def test_vocabulary_is_a_constant() -> None:
    """No data pass, no sample size to get wrong.

    ConfSeq's tokenizer splits every multi-character symbol, so its alphabet is
    closed: 361 angle tokens plus 84 single characters. Enumerating it means the
    vocabulary is reproducible from nothing and no molecule can fall out of it.

    The number is pinned because the ids are baked into every token stream and
    into a trained model's embedding table -- changing the alphabet silently
    renumbers a corpus that a checkpoint was trained on.
    """
    v = build_vocab({})
    assert v.confseq.size == 445
    assert v.vocab_size == 12740
    tokens = set(v.confseq.tokens)
    # The class of symbol whose absence cost 63% of a GEOM sample when the
    # alphabet was collected from 396 conformers instead of enumerated.
    assert {"#", "o", "s", "r", "l", "H", "[", "]", "%", "|"} <= tokens
    assert {f"<{d}>" for d in (-180, 0, 180)} <= tokens


def test_observation_cannot_add_to_the_enumeration_unnoticed() -> None:
    """A symbol the enumeration misses must change the vocabulary, not vanish.

    ``counts`` is kept so a corpus pass can check the enumeration rather than
    depend on it: anything it supplies that is not enumerated is a signal that
    the closed-alphabet assumption has broken.
    """
    base = build_vocab({}).confseq.size
    assert build_vocab({"C": 5, "<0>": 9}).confseq.size == base
    assert build_vocab({"[Se]": 1}).confseq.size == base + 1

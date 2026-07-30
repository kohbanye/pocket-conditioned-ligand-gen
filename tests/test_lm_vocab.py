"""Tests for the unified all-atom LM vocabulary."""

from __future__ import annotations

from prolit.tokenizers.lm_vocab import (
    BOS_ID,
    EOS_ID,
    L_CLOSE_ID,
    L_OPEN_ID,
    NUM_SPECIAL,
    P_CLOSE_ID,
    P_OPEN_ID,
    AtomLMVocab,
)


class TestAtomLMVocab:
    def test_vocab_size_and_offset(self) -> None:
        v = AtomLMVocab(codebook_size=100)
        assert v.offset == NUM_SPECIAL
        assert v.vocab_size == NUM_SPECIAL + 100

    def test_build_sequence_structure(self) -> None:
        v = AtomLMVocab(codebook_size=50)
        seq = v.build_sequence([0, 1, 2], [3, 4])
        # <bos><p> p p p </p><l> l l </l><eos>
        assert seq[0] == BOS_ID
        assert seq[1] == P_OPEN_ID
        assert seq[2:5] == [v.offset + 0, v.offset + 1, v.offset + 2]
        assert seq[5] == P_CLOSE_ID
        assert seq[6] == L_OPEN_ID
        assert seq[7:9] == [v.offset + 3, v.offset + 4]
        assert seq[9] == L_CLOSE_ID
        assert seq[10] == EOS_ID

    def test_shared_range_protein_and_ligand(self) -> None:
        # The same code maps to the same id whether protein or ligand.
        v = AtomLMVocab(codebook_size=50)
        seq = v.build_sequence([7], [7])
        # Both occurrences of code 7 produce the same token id.
        ids = [t for t in seq if t >= v.offset]
        assert ids == [v.offset + 7, v.offset + 7]

    def test_round_trip(self) -> None:
        v = AtomLMVocab(codebook_size=64)
        protein = [0, 5, 63, 12]
        ligand = [1, 2, 3]
        seq = v.build_sequence(protein, ligand)
        p_out, l_out = v.split_sequence(seq)
        assert p_out == protein
        assert l_out == ligand

    def test_split_ignores_unknown_and_specials(self) -> None:
        v = AtomLMVocab(codebook_size=10)
        # Manually crafted: stray specials / out-of-range tokens are dropped.
        seq = [BOS_ID, P_OPEN_ID, v.offset + 2, P_CLOSE_ID, L_OPEN_ID, v.offset + 9]
        p_out, l_out = v.split_sequence(seq)
        assert p_out == [2]
        assert l_out == [9]

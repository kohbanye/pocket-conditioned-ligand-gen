"""Unit tests for the complex-token MLM dataset (BERT-style dynamic masking)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from src.data.mlm_dataset import (
    IGNORE_INDEX,
    MLMTokenDataset,
    collate_mlm,
)
from src.tokenizers.lm_vocab import (
    BOS_ID,
    EOS_ID,
    L_CLOSE_ID,
    L_OPEN_ID,
    NUM_SPECIAL,
    P_CLOSE_ID,
    P_OPEN_ID,
    PAD_ID,
)

if TYPE_CHECKING:
    from pathlib import Path

# One complex: <bos><p> 10 11 12 </p><l> 20 21 22 23 </l><eos>
_DOC = [
    BOS_ID,
    P_OPEN_ID,
    10,
    11,
    12,
    P_CLOSE_ID,
    L_OPEN_ID,
    20,
    21,
    22,
    23,
    L_CLOSE_ID,
    EOS_ID,
]
_SPECIALS = {PAD_ID, BOS_ID, EOS_ID, P_OPEN_ID, P_CLOSE_ID, L_OPEN_ID, L_CLOSE_ID}
_ATOM_CODEBOOK = 32
_BASE_VOCAB = NUM_SPECIAL + _ATOM_CODEBOOK  # 39
_MASK_ID = _BASE_VOCAB  # 39


def _write_cache(tmp_path: Path, docs: list[list[int]]) -> tuple[Path, Path]:
    flat = np.concatenate([np.asarray(d, dtype=np.uint16) for d in docs])
    lengths = np.asarray([len(d) for d in docs], dtype=np.uint16)
    bin_path = tmp_path / "train.bin"
    len_path = tmp_path / "train.len"
    flat.tofile(bin_path)
    lengths.tofile(len_path)
    return bin_path, len_path


def _dataset(tmp_path: Path, *, ligand_only: bool = False) -> MLMTokenDataset:
    bin_path, len_path = _write_cache(tmp_path, [_DOC, _DOC])
    return MLMTokenDataset(
        bin_path,
        len_path,
        block_size=64,
        base_vocab_size=_BASE_VOCAB,
        mask_token_id=_MASK_ID,
        mask_prob=0.5,
        ligand_only_masking=ligand_only,
    )


def test_specials_never_masked_and_labels_align(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    assert len(ds) == 2
    original = np.asarray(_DOC, dtype=np.int64)
    for _ in range(50):  # dynamic masking is random; sample many draws
        item = ds[0]
        input_ids = item["input_ids"].numpy()
        labels = item["labels"].numpy()

        # Structure markers are never corrupted or supervised.
        for i, tok in enumerate(_DOC):
            if tok in _SPECIALS:
                assert input_ids[i] == tok
                assert labels[i] == IGNORE_INDEX

        # Every supervised position carries the ORIGINAL token as its label.
        supervised = labels != IGNORE_INDEX
        assert supervised.any()  # >=1 masked
        assert np.array_equal(labels[supervised], original[supervised])

        # Masked inputs are either <mask>, a codebook token, or left unchanged.
        for i in np.flatnonzero(supervised):
            assert input_ids[i] == _MASK_ID or NUM_SPECIAL <= input_ids[i] < _BASE_VOCAB


def test_ligand_only_masking_stays_in_ligand_span(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, ligand_only=True)
    ligand_positions = {7, 8, 9, 10}  # indices of 20,21,22,23 in _DOC
    for _ in range(50):
        labels = ds[0]["labels"].numpy()
        supervised = set(np.flatnonzero(labels != IGNORE_INDEX).tolist())
        assert supervised
        assert supervised <= ligand_positions


def test_collate_pads_and_builds_attention_mask(tmp_path: Path) -> None:
    short = [BOS_ID, L_OPEN_ID, 20, L_CLOSE_ID, EOS_ID]
    bin_path, len_path = _write_cache(tmp_path, [_DOC, short])
    ds = MLMTokenDataset(
        bin_path,
        len_path,
        block_size=64,
        base_vocab_size=_BASE_VOCAB,
        mask_token_id=_MASK_ID,
        mask_prob=0.5,
    )
    batch = collate_mlm([ds[0], ds[1]])
    assert batch["input_ids"].shape == (2, len(_DOC))
    # Padded row: real tokens flagged, padding zeroed + ignored.
    assert batch["attention_mask"][1].tolist() == [1] * len(short) + [0] * (
        len(_DOC) - len(short)
    )
    pad_region = slice(len(short), len(_DOC))
    assert torch.all(batch["input_ids"][1][pad_region] == PAD_ID)
    assert torch.all(batch["labels"][1][pad_region] == IGNORE_INDEX)


def test_block_size_truncation(tmp_path: Path) -> None:
    bin_path, len_path = _write_cache(tmp_path, [_DOC])
    ds = MLMTokenDataset(
        bin_path,
        len_path,
        block_size=6,
        base_vocab_size=_BASE_VOCAB,
        mask_token_id=_MASK_ID,
    )
    assert ds[0]["input_ids"].shape[0] == 6

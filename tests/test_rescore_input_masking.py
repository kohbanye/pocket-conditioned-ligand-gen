"""Input masking during head fine-tuning: train-only, side-selectable, in-vocab.

The head has always been fine-tuned on complete complexes, and a 105M encoder
on ~15k of them ends at train/loss 0.018 against val/loss 0.695. Replacing a
share of the codebook tokens with the encoder's ``<mask>`` id is the one input
corruption that stays inside the pretraining distribution, so these tests fix
what it touches: which side is hidden, what the readout is told about it, and
that validation and the corpus are left alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from prolit.config import ProLITMLMConfig, RescoreTrainingConfig
from prolit.data.rescore_dataset import RescoreDataModule, RescoreDataset
from prolit.tokenizers.lm_vocab import (
    BOS_ID,
    EOS_ID,
    L_CLOSE_ID,
    L_OPEN_ID,
    NUM_SPECIAL,
    P_CLOSE_ID,
    P_OPEN_ID,
)

if TYPE_CHECKING:
    from pathlib import Path

N_POCKET, N_LIGAND, N_DOCS = 20, 12, 4
MASK_ID = ProLITMLMConfig().mask_token_id


def _doc(offset: int) -> list[int]:
    pocket = [NUM_SPECIAL + offset + i for i in range(N_POCKET)]
    ligand = [NUM_SPECIAL + offset + 500 + i for i in range(N_LIGAND)]
    return [BOS_ID, P_OPEN_ID, *pocket, P_CLOSE_ID,
            L_OPEN_ID, *ligand, L_CLOSE_ID, EOS_ID]


def _write(tmp_path: Path, split: str = "train") -> Path:
    tokens: list[int] = []
    lengths: list[int] = []
    for d in range(N_DOCS):
        doc = _doc(1000 * d)
        tokens.extend(doc)
        lengths.append(len(doc))
    (tmp_path / f"{split}.bin").write_bytes(
        np.asarray(tokens, dtype=np.uint16).tobytes()
    )
    (tmp_path / f"{split}.len").write_bytes(
        np.asarray(lengths, dtype=np.uint16).tobytes()
    )
    (tmp_path / f"{split}.rmsd").write_bytes(
        np.arange(1, N_DOCS + 1, dtype=np.float32).tobytes()
    )
    (tmp_path / f"{split}.grp").write_bytes(np.zeros(N_DOCS, dtype=np.int32).tobytes())
    return tmp_path


def _dataset(tmp_path: Path, **kw: object) -> RescoreDataset:
    return RescoreDataset(
        tmp_path / "train.bin",
        tmp_path / "train.len",
        tmp_path / "train.rmsd",
        block_size=128,
        group_path=tmp_path / "train.grp",
        **kw,  # ty: ignore[invalid-argument-type]
    )


def _sides(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(pocket slice, ligand slice) of a document laid out by ``_doc``."""
    lo = int(np.flatnonzero(ids == L_OPEN_ID)[0])
    return ids[2 : lo - 1], ids[lo + 1 : lo + 1 + N_LIGAND]


def test_off_by_default(tmp_path: Path) -> None:
    _write(tmp_path)
    ids = _dataset(tmp_path)[0]["input_ids"].numpy()
    assert not (ids == MASK_ID).any()


def test_masks_only_the_named_side(tmp_path: Path) -> None:
    _write(tmp_path)
    for side, (want_pocket, want_ligand) in (
        ("ligand", (False, True)),
        ("pocket", (True, False)),
        ("both", (True, True)),
    ):
        ds = _dataset(
            tmp_path, input_mask_prob=0.5, input_mask_side=side, mask_token_id=MASK_ID
        )
        pocket_hit = ligand_hit = False
        for _ in range(8):
            pocket, ligand = _sides(ds[0]["input_ids"].numpy())
            pocket_hit |= bool((pocket == MASK_ID).any())
            ligand_hit |= bool((ligand == MASK_ID).any())
        assert pocket_hit == want_pocket, side
        assert ligand_hit == want_ligand, side


def test_rate_sets_how_many_are_hidden(tmp_path: Path) -> None:
    _write(tmp_path)
    ds = _dataset(
        tmp_path, input_mask_prob=0.25, input_mask_side="ligand", mask_token_id=MASK_ID
    )
    for _ in range(8):
        ids = ds[0]["input_ids"].numpy()
        assert int((ids == MASK_ID).sum()) == round(0.25 * N_LIGAND)


def test_keep_and_drop_differ_only_in_the_pooled_mask(tmp_path: Path) -> None:
    _write(tmp_path)
    keep, drop = (
        _dataset(
            tmp_path,
            input_mask_prob=0.25,
            input_mask_side="ligand",
            input_mask_pool=pool,
            mask_token_id=MASK_ID,
            seed=11,
        )[0]
        for pool in ("keep", "drop")
    )
    # Same stream, same seed -> the same positions are hidden.
    assert keep["input_ids"].tolist() == drop["input_ids"].tolist()
    assert int(keep["ligand_mask"].sum()) == N_LIGAND
    assert int(drop["ligand_mask"].sum()) == N_LIGAND - round(0.25 * N_LIGAND)
    # A masked position is never in the dropped mask, and always in the kept one.
    hidden = keep["input_ids"].numpy() == MASK_ID
    assert not drop["ligand_mask"].numpy()[hidden].any()
    assert keep["ligand_mask"].numpy()[hidden].all()


def test_the_corpus_on_disk_is_untouched(tmp_path: Path) -> None:
    _write(tmp_path)
    before = (tmp_path / "train.bin").read_bytes()
    ds = _dataset(
        tmp_path, input_mask_prob=0.5, input_mask_side="both", mask_token_id=MASK_ID
    )
    for _ in range(8):
        ds[0]
    assert (tmp_path / "train.bin").read_bytes() == before


def test_validation_is_never_masked(tmp_path: Path) -> None:
    _write(tmp_path, "train")
    _write(tmp_path, "val")
    config = RescoreTrainingConfig(token_dir=tmp_path, block_size=128)
    config.input_mask_prob = 0.5
    config.input_mask_side = "both"
    dm = RescoreDataModule(config)
    dm.setup()
    assert dm._datasets["train"].input_mask_prob == 0.5  # noqa: SLF001
    assert dm._datasets["val"].input_mask_prob == 0.0  # noqa: SLF001
    val = dm._datasets["val"]  # noqa: SLF001
    assert not (val[0]["input_ids"].numpy() == MASK_ID).any()


def test_masked_ids_leave_the_codebook_range(tmp_path: Path) -> None:
    """So the head's id-derived protein mask drops masked pocket rows itself."""
    _write(tmp_path)
    ds = _dataset(
        tmp_path, input_mask_prob=0.5, input_mask_side="pocket", mask_token_id=MASK_ID
    )
    ids = ds[0]["input_ids"].numpy()
    codebook = ProLITMLMConfig().atom_codebook_size
    assert NUM_SPECIAL + codebook <= MASK_ID
    atoms = (ids >= NUM_SPECIAL) & (ids < NUM_SPECIAL + codebook)
    assert int(atoms.sum()) == N_POCKET + N_LIGAND - round(0.5 * N_POCKET)

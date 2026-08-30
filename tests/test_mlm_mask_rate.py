"""A fixed mask rate teaches infilling; an iterative decoder needs the schedule.

MaskGIT-style decoding starts from a fully masked ligand and commits atoms a few
at a time, so the model is asked to predict at every rate from 100% down. A
model trained only at 15% has never seen the top of that range: decoding a fully
masked ligand with one gave RMSD 7.67 A where the causal model gave 1.06, and it
got 0.000 of the codes right.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from prolit.data.mlm_dataset import IGNORE_INDEX, MLMTokenDataset
from prolit.tokenizers.lm_vocab import (
    BOS_ID,
    L_CLOSE_ID,
    L_OPEN_ID,
    NUM_SPECIAL,
    P_CLOSE_ID,
    P_OPEN_ID,
)

if TYPE_CHECKING:
    from pathlib import Path


def _corpus(
    tmp_path: Path, n_docs: int = 24, n_lig: int = 20
) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    docs, lens = [], []
    for _ in range(n_docs):
        pocket = rng.integers(NUM_SPECIAL, NUM_SPECIAL + 100, 12)
        lig = rng.integers(NUM_SPECIAL, NUM_SPECIAL + 100, n_lig)
        doc = np.concatenate(
            [[BOS_ID, P_OPEN_ID], pocket, [P_CLOSE_ID, L_OPEN_ID], lig, [L_CLOSE_ID]]
        ).astype(np.uint16)
        docs.append(doc)
        lens.append(len(doc))
    (tmp_path / "train.bin").write_bytes(np.concatenate(docs).tobytes())
    (tmp_path / "train.len").write_bytes(np.array(lens, dtype=np.uint16).tobytes())
    return tmp_path / "train.bin", tmp_path / "train.len"


def _rates(tmp_path: Path, **kw: float) -> np.ndarray:
    bin_path, len_path = _corpus(tmp_path)
    ds = MLMTokenDataset(
        bin_path, len_path, block_size=512, base_vocab_size=107,
        mask_token_id=107, ligand_only_masking=True, seed=0, **kw,
    )
    out = []
    for i in range(len(ds)):
        item = ds[i]
        n_masked = int((np.asarray(item["labels"]) != IGNORE_INDEX).sum())
        out.append(n_masked / 20)
    return np.array(out)


def test_the_fixed_rate_is_unchanged_by_default(tmp_path: Path) -> None:
    r = _rates(tmp_path, mask_prob=0.15)
    assert set(np.round(r, 4)) == {0.15}


def test_a_max_spreads_the_rate_over_the_range(tmp_path: Path) -> None:
    r = _rates(tmp_path, mask_prob=0.15, mask_prob_max=1.0)
    assert r.min() >= 0.15 - 1e-9
    assert r.max() <= 1.0 + 1e-9
    assert r.std() > 0.1, "rates should actually vary"
    # The top of the schedule -- a nearly fully masked ligand -- must occur.
    assert (r > 0.8).any()


def test_a_max_at_or_below_the_floor_keeps_the_fixed_rate(tmp_path: Path) -> None:
    assert set(np.round(_rates(tmp_path, mask_prob=0.3, mask_prob_max=0.3), 4)) == {0.3}

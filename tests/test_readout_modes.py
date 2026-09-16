"""The readouts, and the default that must not move.

Three are parameter-free (ligand, balanced, allatoms). ``concat`` is the
exception: it keeps the pocket and ligand means side by side, widening the
head's first layer, which is why it gets its own width test below.

``_pool`` is exercised on an unconstructed module rather than a real one: the
pooling is pure arithmetic over masks, while building the module would stand up
a ~99M-parameter encoder to test none of it.
"""

from types import SimpleNamespace

import pytest
import torch

from prolit.model.rescore_module import READOUTS, ComplexRescoreModule
from prolit.tokenizers.lm_vocab import (
    BOS_ID,
    EOS_ID,
    L_CLOSE_ID,
    L_OPEN_ID,
    NUM_SPECIAL,
    P_CLOSE_ID,
    P_OPEN_ID,
)

CODEBOOK = 8192


def module(readout: str) -> ComplexRescoreModule:
    """A ComplexRescoreModule with only what ``_pool`` reads."""
    m = object.__new__(ComplexRescoreModule)
    m.readout = readout
    m.config = SimpleNamespace(model=SimpleNamespace(atom_codebook_size=CODEBOOK))
    return m


def batch() -> dict[str, torch.Tensor]:
    """One complex: two protein atoms, two ligand atoms, markers and padding."""
    p0, p1 = NUM_SPECIAL, NUM_SPECIAL + 1
    l0, l1 = NUM_SPECIAL + 2, NUM_SPECIAL + 3
    tokens = torch.tensor(
        [
            [BOS_ID, P_OPEN_ID, p0, p1, P_CLOSE_ID],
            [L_OPEN_ID, l0, l1, L_CLOSE_ID, EOS_ID],
        ]
    ).reshape(1, -1)
    tokens = torch.cat([tokens, torch.zeros(1, 1, dtype=tokens.dtype)], dim=1)
    ligand = torch.zeros_like(tokens, dtype=torch.bool)
    ligand[0, 6] = ligand[0, 7] = True
    return {
        "input_ids": tokens,
        "attention_mask": tokens != 0,
        "ligand_mask": ligand,
    }


def states() -> torch.Tensor:
    hs = torch.zeros(1, 11, 2)
    hs[0, 2], hs[0, 3] = torch.tensor([1.0, 0.0]), torch.tensor([3.0, 0.0])
    hs[0, 6], hs[0, 7] = torch.tensor([0.0, 5.0]), torch.tensor([0.0, 9.0])
    # Markers and padding carry values that must never reach a mean.
    hs[0, 0], hs[0, 4], hs[0, 10] = 100.0, 100.0, 100.0
    return hs


def test_ligand_is_the_default_and_is_unchanged() -> None:
    b, hs = batch(), states()
    assert module("ligand").readout == "ligand"
    pooled = module("ligand")._pool(hs, b)  # noqa: SLF001
    m = b["ligand_mask"].unsqueeze(-1).to(hs.dtype)
    expected = (hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    torch.testing.assert_close(pooled, expected)
    torch.testing.assert_close(pooled, torch.tensor([[0.0, 7.0]]))


def test_balanced_weights_the_two_sides_equally() -> None:
    pooled = module("balanced")._pool(states(), batch())  # noqa: SLF001
    # protein mean (2,0); ligand mean (0,7); their average.
    torch.testing.assert_close(pooled, torch.tensor([[1.0, 3.5]]))


def test_allatoms_weights_every_atom_equally() -> None:
    pooled = module("allatoms")._pool(states(), batch())  # noqa: SLF001
    # (1+3+0+0)/4 and (0+0+5+9)/4 -- markers and padding excluded.
    torch.testing.assert_close(pooled, torch.tensor([[1.0, 3.5]]))


def test_concat_keeps_the_two_sides_separate() -> None:
    pooled = module("concat")._pool(states(), batch())  # noqa: SLF001
    # protein mean (2,0) then ligand mean (0,7) -- side by side, not averaged.
    torch.testing.assert_close(pooled, torch.tensor([[2.0, 0.0, 0.0, 7.0]]))


@pytest.mark.parametrize("readout", ["ligand", "balanced", "allatoms", "concat"])
def test_head_width_follows_the_readout(readout: str) -> None:
    """Only concat widens the head; a small encoder keeps this cheap."""
    from prolit.config import ProLITMLMConfig, RescoreTrainingConfig  # noqa: PLC0415

    h = 16
    cfg = RescoreTrainingConfig(
        model=ProLITMLMConfig(hidden_size=h, num_hidden_layers=1, num_attention_heads=2)
    )
    first = ComplexRescoreModule(cfg, readout=readout).head[0]
    assert first.in_features == (2 * h if readout == "concat" else h)
    assert first.out_features == h


def test_component_masks_exclude_markers_specials_and_padding() -> None:
    protein, ligand = module("balanced")._component_masks(batch())  # noqa: SLF001
    assert protein.tolist() == [
        [False, False, True, True, False, False, False, False, False, False, False]
    ]
    assert ligand.tolist() == batch()["ligand_mask"].tolist()


def test_a_complex_with_no_protein_atom_is_an_error() -> None:
    b = batch()
    b["attention_mask"] = b["ligand_mask"].clone()
    with pytest.raises(ValueError, match="no protein or no ligand atom"):
        module("allatoms")._pool(states(), b)  # noqa: SLF001


def test_unknown_readout_is_rejected_at_construction() -> None:
    from prolit.config import RescoreTrainingConfig  # noqa: PLC0415

    with pytest.raises(ValueError, match="unknown readout"):
        ComplexRescoreModule(RescoreTrainingConfig(), readout="ligand_max")
    assert set(READOUTS) == {"ligand", "balanced", "allatoms", "concat"}

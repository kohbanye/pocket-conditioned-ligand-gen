"""M14 keeps parameters and checkpoint mode while changing only fixed pooling."""

import sys
from pathlib import Path

import lightning
import pytest
import torch
from torch import nn

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "experiments/affinity_loop_20260914")
)
from m14_readout import M14ReadoutModule, component_masks, pool_components
from prolit.config import RescoreTrainingConfig
from prolit.model.rescore_module import ComplexRescoreModule


def batch():
    tokens = torch.tensor([[1, 3, 7, 8, 4, 5, 9, 6, 2, 0]])
    return dict(input_ids=tokens, attention_mask=tokens != 0, ligand_mask=tokens == 9)


def test_pool_formulas_and_component_gradients():
    b = batch()
    h = torch.arange(10, dtype=torch.float32).view(1, 10, 1).requires_grad_()
    assert pool_components(h, b, "ligand").item() == 6
    assert pool_components(h, b, "balanced").item() == 4.25
    assert pool_components(h, b, "allatoms").item() == pytest.approx(11 / 3)
    for mode, weights in [
        ("balanced", [0.25, 0.25, 0.5]),
        ("allatoms", [1 / 3, 1 / 3, 1 / 3]),
    ]:
        h.grad = None
        pool_components(h, b, mode).sum().backward()
        torch.testing.assert_close(h.grad[0, [2, 3, 6], 0], torch.tensor(weights))
        assert h.grad[0, [0, 1, 4, 5, 7, 8, 9], 0].abs().sum() == 0


@pytest.mark.parametrize("bad", ["zeroP", "zeroL", "mask", "markers"])
def test_invalid_component_masks_rejected(bad):
    b = batch()
    if bad == "zeroP":
        b["attention_mask"][0, 2:4] = False
    if bad == "zeroL":
        b["attention_mask"][0, 6] = False
    if bad == "mask":
        b["ligand_mask"][0, 5] = True
    if bad == "markers":
        b["input_ids"][0, 4] = 3
    with pytest.raises(ValueError):
        component_masks(b)


def test_parameter_and_ligand_numerical_identity(monkeypatch):
    monkeypatch.setattr(
        "prolit.model.rescore_module.build_prolit_mlm", lambda _: nn.Identity()
    )
    cfg = RescoreTrainingConfig()
    torch.manual_seed(7)
    base = ComplexRescoreModule(cfg)
    for mode in ["ligand", "balanced", "allatoms"]:
        torch.manual_seed(7)
        model = M14ReadoutModule(cfg, readout_mode=mode)
        assert sum(p.numel() for p in model.head.parameters()) == 592897
        assert model.state_dict().keys() == base.state_dict().keys()
        for k, v in base.state_dict().items():
            torch.testing.assert_close(v, model.state_dict()[k], rtol=0, atol=0)
        assert model.configure_optimizers.__func__ is base.configure_optimizers.__func__
    for dtype in [torch.float32, torch.bfloat16]:
        h = torch.randn(1, 10, 768).to(dtype)
        actual = M14ReadoutModule(cfg, readout_mode="ligand")._pool(h, batch())
        torch.testing.assert_close(actual, base._pool(h, batch()), rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["ligand", "balanced", "allatoms"])
def test_checkpoint_mode_round_trip(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(
        "prolit.model.rescore_module.build_prolit_mlm", lambda _: nn.Identity()
    )
    module = M14ReadoutModule(RescoreTrainingConfig(), readout_mode=mode)
    path = tmp_path / "model.ckpt"
    checkpoint = {
        "state_dict": module.state_dict(),
        "hyper_parameters": dict(module.hparams),
        "pytorch-lightning_version": lightning.__version__,
    }
    torch.save(checkpoint, path)
    restored = M14ReadoutModule.load_from_checkpoint(path, weights_only=False)
    assert restored.readout_mode == mode
    h = torch.randn(1, 10, 768)
    torch.testing.assert_close(
        module._pool(h, batch()), restored._pool(h, batch()), rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="missing or differs"):
        M14ReadoutModule(
            RescoreTrainingConfig(),
            readout_mode="balanced" if mode != "balanced" else "ligand",
        ).on_load_checkpoint(checkpoint)
    with pytest.raises(ValueError):
        M14ReadoutModule(RescoreTrainingConfig())

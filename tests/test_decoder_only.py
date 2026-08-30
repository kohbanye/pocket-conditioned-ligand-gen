"""Decoder-only training must leave the codes exactly where they were.

That is the whole point of it: the aromatic head is a decoder head, and fixing
it by retraining the tokenizer would move the codes and invalidate every token
stream and language model built on them. If the codes move, this is not a
cheaper fix -- it is the expensive one wearing a flag.
"""

from __future__ import annotations

import torch

from prolit.config import AtomVQVAEConfig, AtomVQVAETrainingConfig
from prolit.model.vqvae_module import AtomVQVAEModule
from prolit.tokenizers.atom import ATOM_DESCRIPTOR_DIM


def _module(*, freeze: bool) -> AtomVQVAEModule:
    atom = AtomVQVAEConfig(
        codebook_size=64, latent_dim=8, hidden_dim=16,
        num_transformer_layers=1, num_attention_heads=2
    )
    atom.freeze_encoder = freeze
    module = AtomVQVAEModule(AtomVQVAETrainingConfig(atom=atom))
    # ``on_fit_start`` also pulls normalization stats off the Trainer, which a
    # bare test has none of; the freezing itself is what is under test.
    if freeze:
        module._freeze_encoder()  # noqa: SLF001
    return module


def _batch(n: int = 12) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(1, n, ATOM_DESCRIPTOR_DIM, generator=generator).abs()


def test_freezing_holds_the_encoder_and_codebook_in_eval() -> None:
    module = _module(freeze=True)
    module.train()          # Lightning does this at every epoch boundary
    for name in AtomVQVAEModule._ENCODER_PARTS:  # noqa: SLF001
        part = getattr(module.vqvae, name, None)
        if part is None:
            continue
        assert not part.training, f"{name} is still in train mode"
        assert not any(p.requires_grad for p in part.parameters()), name


def test_the_decoder_still_trains() -> None:
    module = _module(freeze=True)
    trainable = {
        n for n, p in module.vqvae.named_parameters() if p.requires_grad
    }
    assert trainable, "nothing left to train"
    assert any(n.startswith("recon_head_modules") for n in trainable)
    assert not any(n.startswith("transformer_encoder") for n in trainable)


def test_the_codes_do_not_move_under_a_decoder_step() -> None:
    module = _module(freeze=True)
    module.train()
    x = _batch()
    before = module.vqvae.encode(x[0]).clone()
    codebook = module.vqvae.codebook.embedding.data.clone()

    # A decoder update, done by hand so the test does not need a Trainer.
    params = [p for p in module.vqvae.parameters() if p.requires_grad]
    optimiser = torch.optim.SGD(params, lr=0.1)
    out = module.vqvae(x, torch.ones(1, x.shape[1], dtype=torch.bool))
    loss = sum(out["head_losses"].values())
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    assert torch.equal(module.vqvae.codebook.embedding.data, codebook)
    assert torch.equal(module.vqvae.encode(x[0]), before)


def test_without_the_flag_everything_trains() -> None:
    module = _module(freeze=False)
    assert all(p.requires_grad for p in module.vqvae.transformer_encoder.parameters())

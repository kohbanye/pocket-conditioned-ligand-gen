"""Integration: atom shard -> AtomShardedDataset -> collate -> atom VQ-VAE."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from prolit.config import AtomVQVAEConfig
from prolit.data.atom_descriptors import AtomShardedDataset
from prolit.data.descriptors import collate_molecules
from prolit.tokenizers.atom import LigandAtomDescriptor, ProteinAtomDescriptor
from prolit.tokenizers.descriptor_schema import ATOM_DESCRIPTOR_DIM
from prolit.tokenizers.protein import PocketAtomData
from prolit.tokenizers.vqvae import TransformerVQVAE

if TYPE_CHECKING:
    from pathlib import Path


def _toy_entry(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    atoms = [
        ("C", 0.0, 0.0, 0.0),
        ("C", 1.5, 0.0, 0.0),
        ("O", 2.2, 1.1, 0.0),
        ("N", -1.0, 1.0, 0.5),
    ]
    bonds = [(0, 1, 1), (1, 2, 2), (0, 3, 1)]
    frame = (rng.normal(size=3), np.eye(3))
    lig_desc, elements, _ = LigandAtomDescriptor().compute(atoms, bonds, frame)

    names = ["N", "CA", "C", "O", "CB", "N", "CA", "C", "O", "CB"]
    elems = ["N", "C", "C", "O", "C", "N", "C", "C", "O", "C"]
    coords = rng.normal(size=(10, 3)).astype(np.float32)
    pocket = PocketAtomData(
        ca_coords=coords[[1, 6]],
        atom_coords=coords,
        atom_elements=elems,
        atom_names=names,
        atom_aa=["A"] * 5 + ["G"] * 5,
        atom_chain=["A"] * 10,
        atom_resseq=[1] * 5 + [2] * 5,
        residue_ids=[("A", 1), ("A", 2)],
        pocket_seq="AG",
    )
    prot_desc, _ = ProteinAtomDescriptor().compute(pocket, {}, frame)
    return {
        "protein": prot_desc,
        "ligand": lig_desc,
        "elements": elements,
        "pair_idx": seed,
    }


def test_stream_collate_and_vqvae_forward(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    entries = [_toy_entry(0), _toy_entry(1)]
    torch.save(entries, shard_dir / "shard_0000.pt")

    mean = np.zeros(ATOM_DESCRIPTOR_DIM, dtype=np.float32)
    std = np.ones(ATOM_DESCRIPTOR_DIM, dtype=np.float32)
    ds = AtomShardedDataset(shard_dir, [(0, [0, 1])], mean, std, shuffle=False)

    # Each entry yields a protein AND a ligand sequence -> 4 items, all 33-D.
    items = list(ds)
    assert len(ds) == 4
    assert len(items) == 4
    for seq in items:
        assert seq.shape[1] == ATOM_DESCRIPTOR_DIM

    padded, mask = collate_molecules(items)
    assert padded.shape[0] == 4
    assert padded.shape[2] == ATOM_DESCRIPTOR_DIM

    config = AtomVQVAEConfig(
        hidden_dim=32,
        latent_dim=8,
        codebook_size=32,
        num_transformer_layers=2,
        num_attention_heads=4,
        transformer_feedforward_dim=64,
        max_seq_len=64,
    )
    model = TransformerVQVAE(config)
    model.eval()
    out = model(padded, mask=mask)
    assert out["indices"].shape == mask.shape
    # Real (unmasked) positions get valid codes; padded positions stay -1.
    assert (out["indices"][mask] >= 0).all()
    assert (out["indices"][~mask] == -1).all()
    for name in ("coord", "element", "aa", "bb_sc", "clash"):
        assert name in out["head_losses"]
        assert torch.isfinite(out["head_losses"][name]).all()

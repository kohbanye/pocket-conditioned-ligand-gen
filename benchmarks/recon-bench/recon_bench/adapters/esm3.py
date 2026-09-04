"""ESM3 structure-tokenizer reconstruction (in-process).

Encode a protein backbone with ``ESM3_structure_encoder_v0`` and decode it back
with ``ESM3_structure_decoder_v0``; compare CA atoms. ESM3 has no ligand path.

Weights download from HuggingFace ``biohub/esm3-sm-open-v1`` on first use
(``huggingface-cli login`` + license acceptance required).
"""

from __future__ import annotations

import numpy as np

from recon_bench.adapters.base import ReconstructionModel
from recon_bench.structio import Backbone, read_backbone
from recon_bench.types import ModalityRecon, ReconResult, Sample


def chain_break_layout(
    bb: Backbone,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lay a (possibly multi-chain) backbone out the way ESM3 expects one.

    ESM3 encodes a complex as a *single* sequence carrying one separator residue
    per chain boundary: NaN coordinates, ``residue_index`` -1, and a CHAINBREAK
    structure token -- which is what ``esm/models/esm3.py`` fills in wherever the
    sequence has a ``|`` (see ``ProteinComplex.from_chains``). Residue indices
    are ESM3's own default from ``ProteinChain.from_atom37``: one run of 1..L
    over the whole input.

    Handing ESM3 the chains butt-joined instead, indexed by author residue
    number, breaks it twice over. The decoder folds chain B onto the end of
    chain A because nothing marks the boundary, and author numbering restarts
    per chain, so residue 5 of chain A and residue 5 of chain B reach the
    relative-position embedding at offset 0 -- indistinguishable from a residue
    and itself. On CASP16's 57 two-chain samples that cost ESM3 4.1 A of
    pocket-scope Kabsch RMSD (5.11 -> 1.02) and produced every one of its
    apparent reconstruction outliers.

    Returns ``(coords, residue_index, is_residue, order)``: the encoder inputs,
    a mask marking the rows that are real residues, and the ``bb`` row index of
    each real residue in the order ESM3 sees it.
    """
    coords: list[np.ndarray] = []
    is_residue: list[bool] = []
    order: list[int] = []
    for i, chain in enumerate(dict.fromkeys(bb.chain_ids.tolist())):
        if i:  # separator residue between chains, never scored
            coords.append(np.full((3, 3), np.nan))
            is_residue.append(False)
        for row in np.flatnonzero(bb.chain_ids == chain):
            coords.append(bb.coords[row])
            is_residue.append(True)
            order.append(int(row))

    is_residue_arr = np.asarray(is_residue)
    residue_index = np.arange(1, len(coords) + 1, dtype=np.int64)
    residue_index[~is_residue_arr] = -1
    return np.stack(coords), residue_index, is_residue_arr, np.asarray(order)


class ESM3Adapter(ReconstructionModel):
    name = "esm3"
    can_protein = True
    can_ligand = False

    def __init__(self, device: str | None = None, **_: object) -> None:
        self.device = device
        self._encoder = None
        self._decoder = None
        self._special = None

    def setup(self) -> None:
        if self._encoder is not None:
            return
        import torch
        from esm.pretrained import (
            ESM3_structure_decoder_v0,
            ESM3_structure_encoder_v0,
        )
        from esm.utils.constants import esm3 as C

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._encoder = ESM3_structure_encoder_v0(self.device)
        self._decoder = ESM3_structure_decoder_v0(self.device)
        self._special = C.VQVAE_SPECIAL_TOKENS

    def reconstruct(self, sample: Sample) -> ReconResult:
        import torch

        if sample.protein_pdb is None:
            return ReconResult(
                self.name, sample.sample_id, ok=False, error="no protein_pdb"
            )
        self.setup()
        bb = read_backbone(sample.protein_pdb, chain=sample.chain)
        coords_np, residue_index_np, is_residue, order = chain_break_layout(bb)

        coords = torch.from_numpy(coords_np).float().unsqueeze(0).to(self.device)
        residue_index = (
            torch.from_numpy(residue_index_np).long().unsqueeze(0).to(self.device)
        )
        breaks = torch.from_numpy(np.flatnonzero(~is_residue)).to(self.device)
        with torch.no_grad():
            _, tokens = self._encoder.encode(coords, residue_index=residue_index)
            if breaks.numel():
                tokens[:, breaks] = self._special["CHAINBREAK"]
            tokens = torch.nn.functional.pad(tokens, (1, 1), value=0)
            tokens[:, 0] = self._special["BOS"]
            tokens[:, -1] = self._special["EOS"]
            out = self._decoder.decode(tokens)
        bb_pred = out["bb_pred"][0, 1:-1].detach().cpu().numpy()  # (L, 3, 3) N,CA,C
        bb_pred = bb_pred[is_residue]  # drop the separators

        ref_ca = bb.ca[order]
        rec_ca = bb_pred[:, 1, :]
        n = min(len(ref_ca), len(rec_ca))
        res_keys = [
            (str(bb.chain_ids[r]), int(bb.res_ids[r])) for r in order[:n]
        ]
        modality = ModalityRecon(
            modality="protein_backbone",
            ref=ref_ca[:n].astype(np.float64),
            rec=rec_ca[:n].astype(np.float64),
            atom_kind="CA",
            n_residues=int(n),
            # One structure token per residue, plus the chain-break tokens ESM3
            # spends on the boundaries.
            n_tokens=int(n + breaks.numel()),
            res_keys=res_keys,
        )
        return ReconResult(
            self.name, sample.sample_id, modalities=[modality], extra={"seq": bb.seq}
        )

"""ESM3 structure-tokenizer reconstruction (in-process).

Encode a protein backbone with ``ESM3_structure_encoder_v0`` and decode it back
with ``ESM3_structure_decoder_v0``; compare CA atoms. ESM3 has no ligand path.

Weights download from HuggingFace ``biohub/esm3-sm-open-v1`` on first use
(``huggingface-cli login`` + license acceptance required).
"""

from __future__ import annotations

import sys

import numpy as np

from recon_bench import paths
from recon_bench.adapters.base import ReconstructionModel
from recon_bench.structio import Backbone, read_backbone
from recon_bench.types import ModalityRecon, ReconResult, Sample


def chain_break_layout(
    bb: Backbone,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lay a (possibly multi-chain) backbone out the way ESM3 expects one.

    The layout itself lives in :func:`prolit.tokenizers.esm3_layout.chain_break_layout`;
    this is the ``Backbone``-shaped door onto it. The corpus builder for the
    stapled baseline needs the same layout to cache ESM3 structure tokens, and a
    second copy is a second chance to reintroduce the multi-chain bug this
    function exists to fix. prolit is not a dependency of this environment (ESM3
    pins a fork of transformers), so the module is imported lazily from the
    source tree on the path.

    Returns ``(coords, residue_index, is_residue, order)``: the encoder inputs,
    a mask marking the rows that are real residues, and the ``bb`` row index of
    each real residue in the order ESM3 sees it.
    """
    _ensure_prolit_on_path()
    from prolit.tokenizers.esm3_layout import chain_break_layout as _layout

    return _layout(bb.coords, bb.chain_ids)


def _ensure_prolit_on_path() -> None:
    src = str(paths.OWN_MODEL_WORKDIR / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


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

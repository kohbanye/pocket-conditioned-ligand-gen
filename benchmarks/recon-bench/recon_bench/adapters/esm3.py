"""ESM3 structure-tokenizer reconstruction (in-process).

Encode a protein backbone with ``ESM3_structure_encoder_v0`` and decode it back
with ``ESM3_structure_decoder_v0``; compare CA atoms. ESM3 has no ligand path.

Weights download from HuggingFace ``biohub/esm3-sm-open-v1`` on first use
(``huggingface-cli login`` + license acceptance required).
"""

from __future__ import annotations

import numpy as np

from recon_bench.adapters.base import ReconstructionModel
from recon_bench.structio import read_backbone
from recon_bench.types import ModalityRecon, ReconResult, Sample


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

        coords = torch.from_numpy(bb.coords).float().unsqueeze(0).to(self.device)
        residue_index = (
            torch.from_numpy(bb.res_ids).long().unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            _, tokens = self._encoder.encode(coords, residue_index=residue_index)
            tokens = torch.nn.functional.pad(tokens, (1, 1), value=0)
            tokens[:, 0] = self._special["BOS"]
            tokens[:, -1] = self._special["EOS"]
            out = self._decoder.decode(tokens)
        bb_pred = out["bb_pred"][0, 1:-1].detach().cpu().numpy()  # (L, 3, 3) N,CA,C

        ref_ca = bb.ca
        rec_ca = bb_pred[:, 1, :]
        n = min(len(ref_ca), len(rec_ca))
        res_keys = [
            (str(c), int(r))
            for c, r in zip(bb.chain_ids[:n], bb.res_ids[:n], strict=False)
        ]
        modality = ModalityRecon(
            modality="protein_backbone",
            ref=ref_ca[:n].astype(np.float64),
            rec=rec_ca[:n].astype(np.float64),
            atom_kind="CA",
            n_residues=int(n),
            n_tokens=int(n),  # one structure token per residue
            res_keys=res_keys,
        )
        return ReconResult(
            self.name, sample.sample_id, modalities=[modality], extra={"seq": bb.seq}
        )

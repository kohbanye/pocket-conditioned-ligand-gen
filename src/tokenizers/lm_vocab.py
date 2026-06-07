"""Unified token vocabulary for the pocket-conditioned autoregressive LM.

The VQ-VAE emits per-atom / per-residue codebook indices in two separate
spaces (protein structure, ligand structure). The LM consumes a single flat
vocabulary that interleaves a handful of special tokens with the two codebook
ranges mapped to disjoint id offsets:

    0  <pad>
    1  <bos>
    2  <eos>
    3  <p>     (protein-pocket block open)
    4  </p>    (protein-pocket block close)
    5  <l>     (ligand block open)
    6  </l>    (ligand block close)
    7                .. 7 + Pc - 1            protein structure tokens
    7 + Pc        .. 7 + Pc + Lc - 1          ligand structure tokens

where ``Pc`` / ``Lc`` are the protein / ligand codebook sizes. A single
training sequence for one complex is::

    <bos> <p> P.. </p> <l> L.. </l> <eos>

Only ``<p>...</p><l>...</l>`` is modelled (the ``<s>`` protein-sequence block
from :mod:`src.tokenizers.sequence` is intentionally excluded for now).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import LigandVQVAEConfig, ProteinVQVAEConfig

# --- Special token ids (fixed, independent of codebook sizes) --------------
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
P_OPEN_ID = 3
P_CLOSE_ID = 4
L_OPEN_ID = 5
L_CLOSE_ID = 6
NUM_SPECIAL = 7

SPECIAL_TOKENS: dict[str, int] = {
    "<pad>": PAD_ID,
    "<bos>": BOS_ID,
    "<eos>": EOS_ID,
    "<p>": P_OPEN_ID,
    "</p>": P_CLOSE_ID,
    "<l>": L_OPEN_ID,
    "</l>": L_CLOSE_ID,
}

# Default codebook sizes for the "2x" VQ-VAE checkpoint (run 3dvcbp0h).
DEFAULT_PROTEIN_CODEBOOK_SIZE = 8192
DEFAULT_LIGAND_CODEBOOK_SIZE = 4096


@dataclass(frozen=True)
class LMVocab:
    """Flat LM vocabulary derived from the two VQ-VAE codebook sizes."""

    protein_codebook_size: int = DEFAULT_PROTEIN_CODEBOOK_SIZE
    ligand_codebook_size: int = DEFAULT_LIGAND_CODEBOOK_SIZE

    @property
    def protein_offset(self) -> int:
        return NUM_SPECIAL

    @property
    def ligand_offset(self) -> int:
        return NUM_SPECIAL + self.protein_codebook_size

    @property
    def vocab_size(self) -> int:
        return NUM_SPECIAL + self.protein_codebook_size + self.ligand_codebook_size

    # -- id-space mapping ---------------------------------------------------

    def protein_token(self, code: int) -> int:
        return self.protein_offset + code

    def ligand_token(self, code: int) -> int:
        return self.ligand_offset + code

    def build_sequence(
        self,
        protein_codes: list[int],
        ligand_codes: list[int],
    ) -> list[int]:
        """Assemble one complex into ``<bos><p>..</p><l>..</l><eos>`` token ids."""
        seq = [BOS_ID, P_OPEN_ID]
        po = self.protein_offset
        seq.extend(po + c for c in protein_codes)
        seq.append(P_CLOSE_ID)
        seq.append(L_OPEN_ID)
        lo = self.ligand_offset
        seq.extend(lo + c for c in ligand_codes)
        seq.append(L_CLOSE_ID)
        seq.append(EOS_ID)
        return seq

    def split_sequence(self, tokens: list[int]) -> tuple[list[int], list[int]]:
        """Inverse of :meth:`build_sequence`: recover protein/ligand codebook indices.

        Special tokens are dropped; protein/ligand ranges are de-offset back to
        their original codebook index space. Tokens outside the known ranges
        are ignored (robust to partially generated sequences).
        """
        protein_codes: list[int] = []
        ligand_codes: list[int] = []
        lo = self.ligand_offset
        po = self.protein_offset
        lig_end = lo + self.ligand_codebook_size
        for tok in tokens:
            if po <= tok < lo:
                protein_codes.append(tok - po)
            elif lo <= tok < lig_end:
                ligand_codes.append(tok - lo)
        return protein_codes, ligand_codes

    @classmethod
    def from_configs(
        cls,
        protein: ProteinVQVAEConfig,
        ligand: LigandVQVAEConfig,
    ) -> LMVocab:
        return cls(
            protein_codebook_size=protein.codebook_size,
            ligand_codebook_size=ligand.codebook_size,
        )


DEFAULT_VOCAB = LMVocab()

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

Only ``<p>...</p><l>...</l>`` is modelled; a ``<s>`` protein-sequence block
is reserved in the vocabulary but nothing emits one.
"""

from __future__ import annotations

from dataclasses import dataclass

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

# Default codebook size for the unified all-atom VQ-VAE (single codebook).
DEFAULT_ATOM_CODEBOOK_SIZE = 8192


@dataclass(frozen=True)
class AtomLMVocab:
    """Flat LM vocabulary over a SINGLE all-atom codebook range.

    The unified all-atom VQ-VAE emits one code space for both protein and
    ligand atoms, so protein and ligand codes share one id range::

        0  <pad>  1  <bos>  2  <eos>
        3  <p>    4  </p>   5  <l>   6  </l>
        7 .. 7 + C - 1      atom structure tokens (protein AND ligand)

    The ``<p>...</p><l>...</l>`` markers (LM-only) still delimit the
    conditioning block from the generated block; the codebook itself does not
    distinguish source, so :meth:`split_sequence` splits by marker, not by id
    range.
    """

    codebook_size: int = DEFAULT_ATOM_CODEBOOK_SIZE

    @property
    def offset(self) -> int:
        return NUM_SPECIAL

    @property
    def vocab_size(self) -> int:
        return NUM_SPECIAL + self.codebook_size

    def atom_token(self, code: int) -> int:
        return self.offset + code

    def build_sequence(
        self,
        protein_codes: list[int],
        ligand_codes: list[int],
    ) -> list[int]:
        """Assemble ``<bos><p>..</p><l>..</l><eos>`` with one shared code range."""
        o = self.offset
        seq = [BOS_ID, P_OPEN_ID]
        seq.extend(o + c for c in protein_codes)
        seq.append(P_CLOSE_ID)
        seq.append(L_OPEN_ID)
        seq.extend(o + c for c in ligand_codes)
        seq.append(L_CLOSE_ID)
        seq.append(EOS_ID)
        return seq

    def split_sequence(self, tokens: list[int]) -> tuple[list[int], list[int]]:
        """Recover protein/ligand codes by ``<p>``/``<l>`` markers (ranges coincide)."""
        protein_codes: list[int] = []
        ligand_codes: list[int] = []
        o = self.offset
        hi = o + self.codebook_size
        mode: str | None = None
        for tok in tokens:
            if tok == P_OPEN_ID:
                mode = "p"
            elif tok == L_OPEN_ID:
                mode = "l"
            elif tok in (P_CLOSE_ID, L_CLOSE_ID):
                mode = None
            elif o <= tok < hi:
                code = tok - o
                if mode == "p":
                    protein_codes.append(code)
                elif mode == "l":
                    ligand_codes.append(code)
        return protein_codes, ligand_codes

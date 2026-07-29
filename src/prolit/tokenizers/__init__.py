"""Tokenization of protein-ligand complexes into ProLIT interface tokens.

Deliberately re-exports only the descriptor schema. Importing the encoders or
the VQ-VAE here would pull torch in at package-import time and, because
:mod:`prolit.tokenizers.separate_vqvae` reads its config from :mod:`prolit.config`
while ``prolit.config`` imports this schema, would close an import cycle. Import
the heavy pieces from their own modules:

    from prolit.tokenizers.atom import LigandAtomDescriptor, ProteinAtomDescriptor
    from prolit.tokenizers.vqvae import TransformerVQVAE
    from prolit.tokenizers.separate_vqvae import SeparateVQVAE
    from prolit.tokenizers.lm_vocab import AtomLMVocab
"""

from prolit.tokenizers.descriptor_schema import (
    ATOM_DESCRIPTOR_DIM,
    ATOM_LAYOUT,
    SOURCE_LIGAND_IDX,
    SOURCE_PROTEIN_IDX,
    fields_by_name,
)

__all__ = [
    "ATOM_DESCRIPTOR_DIM",
    "ATOM_LAYOUT",
    "SOURCE_LIGAND_IDX",
    "SOURCE_PROTEIN_IDX",
    "fields_by_name",
]

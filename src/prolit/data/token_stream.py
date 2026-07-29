"""Turn batches of descriptors into written token streams.

Every corpus builder does the same thing once it has descriptors in hand:
normalize them, buffer until a batch is worth a GPU round trip, encode with the
frozen tokenizer, assemble ``<bos><p> ... </p><l> ... </l><eos>``, and append to
the split's ``.bin``/``.len`` pair. Only the *source* of the descriptors differs
-- CrossDocked shards, PLINDER zips, BioLIP tarballs, GEOM pickles.

That loop used to be copy-pasted into each builder (two of them byte-identical),
so a fix to the flush logic had to be made four times and drifted when it
wasn't. It lives here now, and the builders only yield descriptors.

The encoder is anything exposing ``encode_batch(x, mask) -> (B, L)`` long
indices with ``-1`` at padded positions: both the joint
:class:`~prolit.tokenizers.vqvae.TransformerVQVAE` and the ablation's
:class:`~prolit.tokenizers.separate_vqvae.SeparateVQVAE` qualify. Which of the
two you pass decides whether normalization happens here or inside the tokenizer
-- see :class:`ComplexTokenEncoder`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch

from prolit.data.descriptors import collate_molecules

if TYPE_CHECKING:
    from prolit.data.token_io import SplitWriter
    from prolit.tokenizers.lm_vocab import AtomLMVocab


class SupportsEncodeBatch(Protocol):
    """The slice of a tokenizer this module needs."""

    def encode_batch(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode a padded ``(B, L, D)`` batch to ``(B, L)`` codes, ``-1`` padded."""
        ...


class ComplexTokenEncoder:
    """Buffer ``(protein, ligand)`` descriptor rows and flush them as token docs.

    Either side may be absent: pass ``ligand=None`` for a protein-only pocket
    corpus, or use :meth:`add_ligand` for a ligand-only conformer corpus. The
    emitted sequence keeps both blocks regardless, so an empty ``<p></p>`` marks
    a ligand with no pocket -- the same shape the fine-tuning corpora use, which
    is what lets a model pretrained on either be fine-tuned on complexes.

    ``mean``/``std`` are applied here. For the separate-tokenizer arm pass
    identity statistics (zeros/ones): each of its sub-VQs holds its own modality
    statistics and normalizes internally, so normalizing twice would corrupt the
    descriptors.
    """

    def __init__(  # noqa: PLR0913
        self,
        tokenizer: SupportsEncodeBatch,
        vocab: AtomLMVocab,
        mean: np.ndarray,
        std: np.ndarray,
        writers: dict[str, SplitWriter],
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.mean = mean
        self.std = std
        self.writers = writers
        self.batch_size = batch_size
        self.device = device
        self._protein: dict[str, list[torch.Tensor | None]] = {s: [] for s in writers}
        self._ligand: dict[str, list[torch.Tensor | None]] = {s: [] for s in writers}

    # -- input ---------------------------------------------------------
    def add(
        self,
        split: str,
        protein: np.ndarray | None,
        ligand: np.ndarray | None = None,
    ) -> None:
        """Queue one document. Unknown splits are dropped (e.g. a held-out PDB)."""
        if split not in self._protein:
            return
        self._protein[split].append(self._norm(protein))
        self._ligand[split].append(self._norm(ligand))
        if len(self._protein[split]) >= self.batch_size:
            self.flush(split)

    def add_ligand(self, split: str, ligand: np.ndarray) -> None:
        """Queue a ligand-only document (empty pocket block)."""
        self.add(split, None, ligand)

    def _norm(self, arr: np.ndarray | None) -> torch.Tensor | None:
        if arr is None:
            return None
        return torch.from_numpy((arr - self.mean) / self.std).float()

    # -- output --------------------------------------------------------
    def flush(self, split: str) -> None:
        """Encode and write everything queued for one split."""
        proteins = self._protein[split]
        ligands = self._ligand[split]
        if not proteins:
            return
        protein_codes = self._encode_column(proteins)
        ligand_codes = self._encode_column(ligands)
        self.writers[split].write(
            [
                self.vocab.build_sequence(protein_codes[i], ligand_codes[i])
                for i in range(len(proteins))
            ]
        )
        proteins.clear()
        ligands.clear()

    def flush_all(self) -> None:
        """Flush every split; call once the source is exhausted."""
        for split in list(self._protein):
            self.flush(split)

    def _encode_column(self, descs: list[torch.Tensor | None]) -> list[list[int]]:
        """Encode the present rows of one modality, preserving position.

        Absent rows come back as empty code lists, so a batch mixing pocketed and
        pocket-less documents still lines up with its inputs.
        """
        present = [i for i, d in enumerate(descs) if d is not None]
        codes: list[list[int]] = [[] for _ in descs]
        if not present:
            return codes
        x, mask = collate_molecules([descs[i] for i in present])
        idx = self.tokenizer.encode_batch(
            x.to(self.device), mask.to(self.device)
        ).cpu()
        for row, i in enumerate(present):
            codes[i] = idx[row][mask[row]].tolist()
        return codes

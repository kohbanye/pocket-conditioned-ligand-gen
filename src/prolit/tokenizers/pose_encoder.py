"""Encode a protein pocket once, then many ligand poses inside it.

Rescoring, decoy-corpus construction and affinity labelling all ask the same
question of the tokenizer: given one receptor and a pile of candidate ligand
poses, what is the token sequence for each pose? The pocket is identical across
them, so its codes are computed once and reused; only the ligand block changes.

That structure is why this is a class rather than a function. :meth:`setup_pocket`
does the expensive half (parse the receptor, extract pocket atoms, derive the
canonical frame, quantize), and the ligand methods then run per pose.

It lived in an eval script and was imported from there by three corpus builders
and copied into a benchmark -- four call sites reaching across three layers for
one recipe. Everything that encodes a pose should go through this.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from prolit.data.descriptors import collate_molecules
from prolit.tokenizers.atom import (
    LigandAtomDescriptor,
    ProteinAtomDescriptor,
    precompute_receptor_atom_features_from_text,
    rotate_atom_descriptor,
)
from prolit.tokenizers.protein import (
    compute_canonical_frame,
    extract_pocket_atoms_from_candidates,
    precompute_pocket_atom_candidates_from_text,
)

if TYPE_CHECKING:
    from prolit.config import PocketExtractionConfig
    from prolit.tokenizers.lm_vocab import AtomLMVocab

_DEFAULT_BATCH = 64


class PoseEncoder:
    """Fixed-pocket encoder: protein codes once, ligand codes per pose.

    ``tokenizer`` is anything exposing ``encode_batch`` -- the joint VQ-VAE or
    the separate-tokenizer ablation. ``mean``/``std`` are the normalization
    statistics that accompany it; for the separate arm pass identity statistics,
    because each of its sub-VQs normalizes internally.
    """

    def __init__(  # noqa: PLR0913
        self,
        tokenizer: Any,  # noqa: ANN401
        mean: np.ndarray,
        std: np.ndarray,
        vocab: AtomLMVocab,
        device: torch.device,
        pocket_cfg: PocketExtractionConfig,
    ) -> None:
        self.tokenizer = tokenizer
        self.mean = mean
        self.std = std
        self.vocab = vocab
        self.device = device
        self.pocket_cfg = pocket_cfg
        self.prot_desc = ProteinAtomDescriptor()
        self.lig_desc = LigandAtomDescriptor()
        #: Pocket descriptor kept after :meth:`setup_pocket`, so a rotated
        #: re-encoding does not have to re-parse the receptor.
        self.prot_desc_raw: np.ndarray | None = None

    # -- pocket ---------------------------------------------------------
    def setup_pocket(
        self,
        protein_text: str,
        reference_heavy: np.ndarray,
    ) -> tuple[list[int], Any] | None:
        """Extract the pocket around a reference ligand -> ``(codes, frame)``.

        Returns None when the receptor yields no pocket atoms, which happens for
        malformed or ligand-free entries and is not worth raising over: the
        caller skips that complex.
        """
        precomp = precompute_pocket_atom_candidates_from_text(protein_text)
        pocket = extract_pocket_atoms_from_candidates(
            precomp, reference_heavy, self.pocket_cfg
        )
        if pocket is None or pocket.atom_coords.shape[0] == 0:
            return None
        feats = precompute_receptor_atom_features_from_text(protein_text)
        frame = compute_canonical_frame(pocket.ca_coords.astype(np.float64))
        prot_desc, _ = self.prot_desc.compute(pocket, feats, frame)
        if prot_desc.shape[0] == 0:
            return None
        self.prot_desc_raw = prot_desc
        return self._encode(prot_desc), frame

    def pocket_codes_rotated(self, rotation: np.ndarray) -> list[int]:
        """Re-encode the cached pocket descriptor under an extra frame rotation."""
        if self.prot_desc_raw is None:
            msg = "call setup_pocket() before pocket_codes_rotated()"
            raise RuntimeError(msg)
        return self._encode(rotate_atom_descriptor(self.prot_desc_raw, rotation))

    # -- ligands --------------------------------------------------------
    def ligand_seq(
        self,
        protein_codes: list[int],
        mol: dict,
        frame: Any,  # noqa: ANN401
    ) -> list[int] | None:
        """Token sequence for one pose, or None if it has no encodable atoms."""
        lig_desc, _elements, _mask = self.lig_desc.compute(
            mol["atoms"], mol["bonds"], frame
        )
        if lig_desc.shape[0] == 0:
            return None
        return self.vocab.build_sequence(protein_codes, self._encode(lig_desc))

    def ligand_seqs_batch(
        self,
        protein_codes: list[int],
        mols: list[dict],
        frame: Any,  # noqa: ANN401
    ) -> list[list[int] | None]:
        """Encode many poses in one VQ call; one sequence (or None) per mol.

        Per-pose batch-of-one is the bottleneck when scoring thousands of decoys,
        which is the whole reason this exists alongside :meth:`ligand_seq`.

        Deliberately one ``encode_batch`` call for the whole list rather than
        chunking through :meth:`seqs_from_descs`: padding is per batch, so a
        different split is a different (if equivalent) computation, and this path
        produces published numbers.
        """
        descs = self.ligand_descs(mols, frame)
        valid = [(i, d) for i, d in enumerate(descs) if d.shape[0] > 0]
        out: list[list[int] | None] = [None] * len(mols)
        if not valid:
            return out
        tensors = [
            torch.from_numpy((d - self.mean) / self.std).float() for _, d in valid
        ]
        x, mask = collate_molecules(tensors)
        idx = self._encode_batch(x, mask)
        for k, (i, _) in enumerate(valid):
            out[i] = self.vocab.build_sequence(
                protein_codes, idx[k][mask[k]].tolist()
            )
        return out

    def ligand_descs(self, mols: list[dict], frame: Any) -> list[np.ndarray]:  # noqa: ANN401
        """Pre-quantization descriptors, so rotated variants can be derived.

        Rotating a stored descriptor is far cheaper than recomputing RDKit
        features per orientation; see
        :func:`~prolit.tokenizers.atom.rotate_atom_descriptor`.
        """
        return [self.lig_desc.compute(m["atoms"], m["bonds"], frame)[0] for m in mols]

    def seqs_from_descs(
        self,
        protein_codes: list[int],
        descs: list[np.ndarray],
        rotation: np.ndarray | None = None,
        batch_size: int = _DEFAULT_BATCH,
    ) -> list[list[int] | None]:
        """Quantize pose descriptors (optionally rotated) into token sequences."""
        out: list[list[int] | None] = [None] * len(descs)
        valid = [(i, d) for i, d in enumerate(descs) if d.shape[0] > 0]
        for start in range(0, len(valid), batch_size):
            chunk = valid[start : start + batch_size]
            arrays = [
                d if rotation is None else rotate_atom_descriptor(d, rotation)
                for _, d in chunk
            ]
            tensors = [
                torch.from_numpy((a - self.mean) / self.std).float() for a in arrays
            ]
            x, mask = collate_molecules(tensors)
            idx = self._encode_batch(x, mask)
            for k, (i, _) in enumerate(chunk):
                out[i] = self.vocab.build_sequence(
                    protein_codes, idx[k][mask[k]].tolist()
                )
        return out

    # -- internals ------------------------------------------------------
    def _encode(self, desc: np.ndarray) -> list[int]:
        x, mask = collate_molecules(
            [torch.from_numpy((desc - self.mean) / self.std).float()]
        )
        idx = self._encode_batch(x, mask)
        return idx[0][mask[0]].tolist()

    def _encode_batch(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.encode_batch(
            x.to(self.device), mask.to(self.device)
        ).cpu()

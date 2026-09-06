"""Laying a (possibly multi-chain) backbone out the way ESM3 expects one.

ESM3 encodes a complex as a *single* sequence carrying one separator residue per
chain boundary: NaN coordinates, ``residue_index`` -1, and a CHAINBREAK
structure token -- which is what ``esm/models/esm3.py`` fills in wherever the
sequence has a ``|`` (see ``ProteinComplex.from_chains``). Residue indices are
ESM3's own default from ``ProteinChain.from_atom37``: one run of 1..L over the
whole input.

Handing ESM3 the chains butt-joined instead, indexed by author residue number,
breaks it twice over. The decoder folds chain B onto the end of chain A because
nothing marks the boundary, and author numbering restarts per chain, so residue
5 of chain A and residue 5 of chain B reach the relative-position embedding at
offset 0 -- indistinguishable from a residue and itself. On CASP16's 57
two-chain samples that cost ESM3 4.1 A of pocket-scope Kabsch RMSD (5.11 ->
1.02) and produced every one of its apparent reconstruction outliers
(``docs/results/2026-09-04_esm3_chain_handling.md``).

This lives in ``prolit`` because two callers need it: the reconstruction
benchmark, which round-trips structures through ESM3 to score them, and the
corpus builder, which encodes receptors once to cache their structure tokens for
the stapled-baseline language model. A second copy of this layout is a second
chance to reintroduce the bug above under a different name.
"""

from __future__ import annotations

import numpy as np

__all__ = ["chain_break_layout"]


def chain_break_layout(
    coords: np.ndarray,
    chain_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build ESM3's encoder inputs from per-residue backbone coordinates.

    ``coords`` is (L, 3, 3) ordered N, CA, C; ``chain_ids`` is (L,) the chain of
    each residue, in file order. Residues are regrouped by chain -- an
    interleaved file would otherwise produce a boundary per residue -- and one
    separator row is inserted between consecutive chains.

    Returns ``(coords, residue_index, is_residue, order)``: the encoder inputs,
    a mask marking the rows that are real residues, and the input row index of
    each real residue in the order ESM3 sees it. ``order`` is what keeps a
    prediction aligned with the reference it will be scored against.
    """
    out_coords: list[np.ndarray] = []
    is_residue: list[bool] = []
    order: list[int] = []
    for i, chain in enumerate(dict.fromkeys(chain_ids.tolist())):
        if i:  # separator residue between chains, never scored
            out_coords.append(np.full((3, 3), np.nan))
            is_residue.append(False)
        for row in np.flatnonzero(chain_ids == chain):
            out_coords.append(coords[row])
            is_residue.append(True)
            order.append(int(row))

    is_residue_arr = np.asarray(is_residue)
    residue_index = np.arange(1, len(out_coords) + 1, dtype=np.int64)
    residue_index[~is_residue_arr] = -1
    return (
        np.stack(out_coords),
        residue_index,
        is_residue_arr,
        np.asarray(order),
    )

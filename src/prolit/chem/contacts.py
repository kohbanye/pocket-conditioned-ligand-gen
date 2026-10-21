"""Per-ligand-atom protein contact labels, for the MLM's auxiliary heads.

Two labels per ligand heavy atom, both read off coordinates:

* ``residue`` -- the amino-acid class of the nearest protein residue within
  :data:`CONTACT_CUT` (21 classes: the 20 standard one-letter codes plus
  ``OTHER_RESIDUE`` for anything else), or :data:`IGNORE` when nothing is in
  range.
* ``itype`` -- which interaction the atom makes, as one of
  ``none / ionic / hbond-like / hydrophobic``.

**These are a training target, not a scoring criterion.** A placed atom's
contacts are computable exactly, which is what
:func:`~prolit.chem.clash_probe.make_clash_probe` does for the decode loop. The
reason to also predict them is the case geometry cannot cover: what contact a
position *should* make while its atom is still masked. A probe on the plain MLM
reaches 18% on ``residue`` against a 9.3% majority baseline (2026-10-09), so
the information is present but thin; these labels are what an auxiliary
objective would use to thicken it.

**No hydrogens.** This pipeline carries heavy atoms only, so the angular term
every real H-bond definition uses is unavailable and ``hbond-like`` is a
heavy-atom donor/acceptor proximity class. It will call an H-bond where the
geometry forbids one. Named ``hbond_like`` rather than ``hbond`` so a caller
cannot mistake it for the real thing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import KDTree

if TYPE_CHECKING:
    from collections.abc import Sequence

#: One-letter codes, matching ``PocketAtomData.atom_aa``.
RESIDUE_CLASSES: tuple[str, ...] = tuple("ARNDCQEGHILKMFPSTWYV")
RESIDUE_INDEX: dict[str, int] = {a: i for i, a in enumerate(RESIDUE_CLASSES)}
#: Non-standard residue, or one this table does not name.
OTHER_RESIDUE = len(RESIDUE_CLASSES)
NUM_RESIDUE_CLASSES = OTHER_RESIDUE + 1

ITYPE_NONE = 0
ITYPE_IONIC = 1
ITYPE_HBOND_LIKE = 2
ITYPE_HYDROPHOBIC = 3
NUM_ITYPE_CLASSES = 4
ITYPE_NAMES: tuple[str, ...] = ("none", "ionic", "hbond-like", "hydrophobic")

#: Written into both label streams where no label applies -- a ligand-only or
#: protein-only document, a special token, or a padded position. Must match the
#: loss's ignore index so those positions never contribute a gradient.
IGNORE = 255

#: Furthest a protein atom can be and still count as this atom's contact.
CONTACT_CUT = 4.5

#: Distance windows. Taken from the ranges PLIP and ProLIF use, not fitted
#: here: fitting them to our own reference distribution would make the target
#: depend on the tokenizer whose representation we are then testing.
IONIC_RANGE = (2.0, 4.0)
HBOND_RANGE = (2.4, 3.5)
HYDROPHOBIC_RANGE = (3.3, 4.5)

_POLAR = frozenset({"N", "O"})
_APOLAR = frozenset({"C", "S", "F", "CL", "BR", "I"})
#: Formally charged side-chain tips, as (one-letter residue, PDB atom name).
_CHARGED_PROTEIN_ATOMS = frozenset({
    ("D", "OD1"), ("D", "OD2"),
    ("E", "OE1"), ("E", "OE2"),
    ("R", "NE"), ("R", "NH1"), ("R", "NH2"),
    ("K", "NZ"),
    ("H", "ND1"), ("H", "NE2"),
})


def _pair_itype(
    ligand_element: str,
    protein_element: str,
    residue: str,
    atom_name: str,
    distance: float,
) -> int:
    """The interaction one ligand/protein heavy-atom pair makes, if any."""
    le, pe = ligand_element.upper(), protein_element.upper()
    if le in _POLAR and pe in _POLAR:
        charged = (residue, atom_name) in _CHARGED_PROTEIN_ATOMS
        if charged and IONIC_RANGE[0] <= distance <= IONIC_RANGE[1]:
            return ITYPE_IONIC
        if HBOND_RANGE[0] <= distance <= HBOND_RANGE[1]:
            return ITYPE_HBOND_LIKE
    if (
        le in _APOLAR
        and pe in _APOLAR
        and HYDROPHOBIC_RANGE[0] <= distance <= HYDROPHOBIC_RANGE[1]
    ):
        return ITYPE_HYDROPHOBIC
    return ITYPE_NONE


def _heavy_protein(
    coords: np.ndarray,
    elements: Sequence[str],
    residues: Sequence[str],
    atom_names: Sequence[str],
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """The protein side with hydrogens dropped, all four arrays kept in step.

    One mask applied to four parallel sequences: filtering them separately is
    how a residue name ends up attached to the wrong atom, which would be
    invisible in the labels and wrong in the same way for every complex.
    """
    pc = np.asarray(coords, dtype=float)
    pel = [str(e).upper() for e in elements]
    keep = np.array([e != "H" for e in pel])
    if not keep.any():
        return np.empty((0, 3)), [], [], []
    return (
        pc[keep],
        [e for e, k in zip(pel, keep, strict=True) if k],
        [str(r).strip().upper() for r, k in zip(residues, keep, strict=True) if k],
        [str(a).strip().upper() for a, k in zip(atom_names, keep, strict=True) if k],
    )


def contact_labels(  # noqa: PLR0913 -- plain arrays, not a pocket container
    ligand_coords: np.ndarray,
    ligand_elements: Sequence[str],
    protein_coords: np.ndarray,
    protein_elements: Sequence[str],
    protein_residues: Sequence[str],
    protein_atom_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """``(residue, itype)`` as ``uint8`` arrays, one entry per ligand atom.

    Hydrogens are dropped from the protein side; the caller is expected to pass
    ligand heavy atoms, in the order their tokens are emitted, because the
    labels are consumed positionally against the token stream.

    Both arrays are :data:`IGNORE` where nothing lies within
    :data:`CONTACT_CUT`, so "no protein nearby" is never confused with
    :data:`ITYPE_NONE`, which means "close, but making no named interaction".
    """
    n = len(ligand_coords)
    residue = np.full(n, IGNORE, dtype=np.uint8)
    itype = np.full(n, IGNORE, dtype=np.uint8)
    if n == 0 or len(protein_coords) == 0:
        return residue, itype

    pc, pel, pres, pnames = _heavy_protein(
        protein_coords, protein_elements, protein_residues, protein_atom_names
    )
    if len(pc) == 0:
        return residue, itype

    tree = KDTree(pc)
    for i, xyz in enumerate(np.asarray(ligand_coords, dtype=float)):
        near = tree.query_ball_point(xyz, CONTACT_CUT)
        if not near:
            continue
        d = np.linalg.norm(pc[near] - xyz, axis=1)
        order = np.argsort(d)
        nearest = near[order[0]]
        residue[i] = RESIDUE_INDEX.get(pres[nearest], OTHER_RESIDUE)
        # Strongest interaction over every atom in range, not just the nearest:
        # the closest protein atom is often a backbone carbon while the
        # interaction that matters is a side-chain oxygen slightly further out.
        best = ITYPE_NONE
        le = str(ligand_elements[i])
        for k in order:
            j = near[k]
            t = _pair_itype(le, pel[j], pres[j], pnames[j], float(d[k]))
            if t != ITYPE_NONE and (best == ITYPE_NONE or t < best):
                best = t
        itype[i] = best
    return residue, itype


__all__ = [
    "CONTACT_CUT",
    "IGNORE",
    "ITYPE_HBOND_LIKE",
    "ITYPE_HYDROPHOBIC",
    "ITYPE_IONIC",
    "ITYPE_NAMES",
    "ITYPE_NONE",
    "NUM_ITYPE_CLASSES",
    "NUM_RESIDUE_CLASSES",
    "OTHER_RESIDUE",
    "RESIDUE_CLASSES",
    "RESIDUE_INDEX",
    "contact_labels",
]

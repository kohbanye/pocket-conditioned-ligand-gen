"""Which ligand positions sit inside the protein, for a batch of code sequences.

:func:`prolit.model.mlm_decode.refine_codes` with ``order="clash"`` re-decides
the positions whose atoms are in the wall, and picks a replacement from the
model's own top candidates that is not. It needs to ask "would THIS code put the
atom in the wall", many times, cheaply -- and it must not learn how to decode a
code or how to read a receptor to do it.

So the decoding stays with the caller. This takes a ``decode`` callable that
turns a batch of code sequences into world-frame coordinates and radii, and adds
the only thing that is genuinely chemistry: the receptor's van der Waals
neighbourhood, using the same ``CLASH_FRACTION`` the deployed rigid fit and the
benchmark's clash metric both use, so the objective and the metric agree.

The measurement that makes this worth having: on the deployed arm a generated
molecule carries 0.89 clashing terminal heavy atoms and the crystal reference
carries 0.10, and 84% of those atoms had a code available that keeps the element
and the bond and clears the wall.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.spatial import KDTree

from prolit.chem.rigid_fit import CLASH_FRACTION, vdw_radii

if TYPE_CHECKING:
    from collections.abc import Callable

    #: ``(B, n) codes -> ((B, n, 3) world coordinates, (B, n) radii)``.
    DecodeToWorld = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]

    #: What the probe returns: which positions of each sequence are unacceptable.
    ClashProbe = Callable[..., np.ndarray]

__all__ = ["make_clash_probe"]


#: A bond shorter or longer than this, in Angstrom, is not a bond any more.
#: The same window the codebook reachability probe used when it counted, for
#: each atom the model had put in the wall, how many codes would have got it out
#: while keeping the molecule intact.
BOND_RANGE: tuple[float, float] = (1.10, 1.95)

#: The narrowest angle, in degrees, a replacement may leave at or beside the
#: atom it moves. Read off the CONTROL arm's own geometry (its 0.5th percentile
#: is 70.5 degrees), not chosen: the rule is "no worse than the arm this is
#: trying not to damage", and a threshold picked by intuition would either kill
#: the escapes or fail to fix the check.
MIN_ANGLE_DEG: float = 70.0

#: The closest two ligand atoms three or more bonds apart may come, in Angstrom.
#: The control arm's 0.1th percentile is 2.19.
MIN_NONBONDED: float = 2.2


class _Wall(NamedTuple):
    """The receptor, prepared once: the probe is called per candidate code."""

    tree: KDTree
    coords: np.ndarray
    radii: np.ndarray
    reach: float


def _in_wall(p: np.ndarray, r: float, wall: _Wall) -> bool:
    """Is this atom closer than ``CLASH_FRACTION`` of the summed radii to the wall."""
    for j in wall.tree.query_ball_point(p, r + wall.reach):
        if float(np.linalg.norm(p - wall.coords[j])) < CLASH_FRACTION * (
            r + wall.radii[j]
        ):
            return True
    return False


class _Geometry(NamedTuple):
    """The windows a replacement has to stay inside to leave a molecule behind."""

    incident: dict[int, list[int]]
    bond_lo: float
    bond_hi: float
    cos_max: float  # cos of the narrowest allowed angle
    nonbonded_min: float


def _angle_too_narrow(
    pose: np.ndarray, centre: int, arms: list[int], g: _Geometry
) -> bool:
    """Does any pair of ``arms`` close past the allowed angle at ``centre``."""
    for a in range(len(arms)):
        for b in range(a + 1, len(arms)):
            u, v = pose[arms[a]] - pose[centre], pose[arms[b]] - pose[centre]
            n = float(np.linalg.norm(u) * np.linalg.norm(v))
            if n > 0 and float(np.dot(u, v)) / n > g.cos_max:
                return True
    return False


def _local_geometry_broken(pose: np.ndarray, i: int, g: _Geometry) -> bool:
    """Would moving atom ``i`` here stop the molecule being a molecule.

    Three windows, all read off the control arm's own geometry: the bonds at
    ``i`` stay bonds, no angle at ``i`` or at one of its neighbours closes past
    ``MIN_ANGLE_DEG``, and nothing three or more bonds away comes closer than
    ``MIN_NONBONDED``.
    """
    n = pose.shape[0]
    nbrs = [k for k in g.incident.get(i, ()) if k < n]
    for k in nbrs:
        d = float(np.linalg.norm(pose[i] - pose[k]))
        if d < g.bond_lo or d > g.bond_hi:
            return True
    if _angle_too_narrow(pose, i, nbrs, g):
        return True
    for k in nbrs:
        # The angle i-k-x at each neighbour, which moving i also changes.
        arms = [x for x in g.incident.get(k, ()) if x < n]
        if len(arms) > 1 and _angle_too_narrow(pose, k, arms, g):
            return True
    far = set(range(n)) - {i} - set(nbrs)
    for k in nbrs:
        far -= set(g.incident.get(k, ()))
    if far:
        idx = np.fromiter(far, dtype=int, count=len(far))
        if float(np.min(np.linalg.norm(pose[idx] - pose[i], axis=1))) < g.nonbonded_min:
            return True
    return False


def make_clash_probe(  # noqa: PLR0913 -- four independent windows, each measured
    decode: DecodeToWorld,
    receptor_coords: np.ndarray,
    receptor_elements: list[str],
    *,
    bonds: list[tuple[int, int]] | None = None,
    bond_range: tuple[float, float] = BOND_RANGE,
    min_angle_deg: float = MIN_ANGLE_DEG,
    min_nonbonded: float = MIN_NONBONDED,
) -> ClashProbe:
    """A probe answering "which positions of each sequence are unacceptable".

    Without ``bonds`` that means "in the wall", which is the right question for
    choosing WHICH positions to re-decide. With ``bonds`` it also means "and it
    wrecked the local geometry", which is the right question for choosing a
    REPLACEMENT -- and the two must not be confused.

    What counts as wrecked was measured check by check, not guessed. An arm
    whose picker looked only at the wall lost 7.0 points of PoseBusters, and the
    per-check breakdown says where they went: bond lengths 6.2% -> 12.2%, bond
    angles 10.3% -> 17.5%, internal steric clash 7.2% -> 11.2%. Constraining
    bond length alone brought lengths back (12.2% -> 7.5%) and recovered only
    2.2 of the 7.0 points, because angles and internal clashes are the larger
    half. All three are checked here, each against a window read off the control
    arm's own distribution.

    The receptor is read once and its KD-tree is built once, because the probe
    is called on every candidate code of every masked position of every round.
    Hydrogens are dropped: the clash criterion is calibrated on heavy atoms.
    """
    coords = np.asarray(receptor_coords, dtype=float)
    keep = np.array([e != "H" for e in receptor_elements])
    if keep.any():
        coords = coords[keep]
        receptor_elements = [
            e for e, k in zip(receptor_elements, keep, strict=True) if k
        ]
    radii = vdw_radii(list(receptor_elements))
    wall = _Wall(KDTree(coords), coords, radii, float(radii.max()))

    incident: dict[int, list[int]] = {}
    if bonds:
        for u, v in bonds:
            incident.setdefault(u, []).append(v)
            incident.setdefault(v, []).append(u)
    geom = _Geometry(incident, *bond_range, np.cos(np.radians(min_angle_deg)),
                     min_nonbonded)

    def probe(seqs: np.ndarray, only: int | None = None) -> np.ndarray:
        """``only`` restricts the answer to one column, left False elsewhere.

        The picker asks about the one position it is replacing, and evaluating
        all ~23 of them per candidate code is that many times the work for an
        answer it discards.
        """
        seqs = np.asarray(seqs)
        world, lig_radii = decode(seqs)
        out = np.zeros(seqs.shape, dtype=bool)
        cols = range(world.shape[1]) if only is None else (only,)
        for b in range(world.shape[0]):
            for i in cols:
                out[b, i] = _in_wall(world[b, i], float(lig_radii[b, i]), wall) or (
                    bool(incident) and _local_geometry_broken(world[b], i, geom)
                )
        return out

    return probe

"""Slide the whole ligand off the receptor wall, without bending it.

The decoder's pose error is **global, not per-atom**. Measured on generated
ligands that clash: optimising six rigid degrees of freedom, bounded to 2.5 A
of translation and 30 degrees of rotation, clears every clash in 92.5% of them
and removes 86.3% of all clashing atoms, at a median translation of 1.66 A. The
same procedure applied to crystal ligands displaced by a known 2.0 A recovers
100% of them, which is what says the search itself is not the thing being
measured.

That number is the whole justification for this module. A per-atom error could
not be undone by a rigid transform; this one can, so the defect is that the
language model puts a chemically fine molecule in slightly the wrong place --
consistent with the rigid-displacement calibration in
``docs/results/2026-08-19_generation_97targets.md``, where displacing crystal
ligands by 2.0 A reproduces the model's clash statistics.

**This is not the flexible pocket-aware relaxation that**
:mod:`prolit.chem.relax` **records as a failure.** That one relieved clashes by
bending the ligand and paid for it in PoseBusters geometry (0.932 -> 0.851).
A rigid transform cannot bend anything: every bond length, every angle, every
torsion, and the internal-clash check are invariant under it, so the
intramolecular half of PoseBusters is unchanged *by construction* and only the
protein-distance check can move -- in one direction.

The objective is the summed squared van der Waals overlap, i.e. the same
quantity the benchmark's clash count thresholds. It is deliberately **not** a
docking score: the model must not be tuned on the function it is scored by. And
it carries no weights to balance -- one objective, two bounds, both set by what
was measured rather than tuned:

* ``max_translation`` 2.5 A -- above the 1.66 A the fix actually needs, and
  above the 2.0 A rigid displacement that reproduces the model's clash rate.
* ``max_rotation`` 30 degrees -- enough to unwind a mis-set ring plane, small
  enough that the pose stays the pose the pocket conditioned.

Both are hard bounds, not penalties, so nothing here trades off against
anything else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from collections.abc import Callable

#: Van der Waals radii in A (Bondi), for the elements the tokenizer emits.
VDW_RADII: dict[str, float] = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "F": 1.47, "Cl": 1.75, "Br": 1.85, "I": 1.98, "B": 1.92, "Si": 2.10,
}
_DEFAULT_RADIUS = 1.70

#: The radii AutoDock Vina scores with (its "X-S" set), which run about 0.2 A
#: larger per atom than Bondi -- 0.4 A per contact.
#:
#: This matters because the objective below is zero the moment nothing overlaps
#: *by its own radii*, and its gradient dies there. Optimised against Bondi, a
#: pose stops exactly on the Bondi contact surface, which sits 0.4 A inside
#: Vina's. Measured over 99 targets: 59% of generated ligand atoms overlap the
#: receptor by Vina's measure (reference ligands 20%), the median surface gap is
#: -0.467 A against the reference's -0.104, and Vina's repulsion term is 7.50
#: against 1.64 for FLOWR -- while every *attractive* term is better than
#: FLOWR's. The molecules are in the right place; they are pressed 0.4 A too
#: deep, and the fitter thinks it is finished.
VINA_RADII: dict[str, float] = {
    "C": 1.90, "N": 1.80, "O": 1.70, "S": 2.00, "P": 2.10,
    "F": 1.50, "Cl": 1.80, "Br": 2.00, "I": 2.20, "H": 1.20,
}
_DEFAULT_VINA_RADIUS = 1.90
#: Below this a rotation vector is the identity to floating-point precision.
_NEGLIGIBLE = 1e-9

#: A contact closer than this fraction of the summed radii counts as a clash.
#: The benchmark's own threshold, so the objective and the metric agree.
CLASH_FRACTION = 0.75


def vdw_radii(elements: list[str], *, scoring: bool = False) -> np.ndarray:
    """Radii for ``elements``, falling back to carbon for anything unlisted.

    ``scoring=True`` returns the larger set AutoDock Vina scores with, so a
    pose relieved against them lands on the surface Vina rewards rather than
    0.4 A inside it. See :data:`VINA_RADII`.
    """
    if scoring:
        return np.array([VINA_RADII.get(e, _DEFAULT_VINA_RADIUS) for e in elements])
    return np.array([VDW_RADII.get(e, _DEFAULT_RADIUS) for e in elements])


def _rotation(axis_angle: np.ndarray) -> np.ndarray:
    """Rodrigues' formula: a rotation vector to a rotation matrix."""
    theta = float(np.linalg.norm(axis_angle))
    if theta < _NEGLIGIBLE:
        return np.eye(3)
    k = axis_angle / theta
    cross = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)


def _clamp(
    params: np.ndarray, max_translation: float, max_rotation: float
) -> tuple[np.ndarray, np.ndarray]:
    """Project the six parameters back inside the bounds.

    Clamping inside the objective rather than handing bounds to the optimiser
    keeps the search unconstrained -- Powell walks a smooth function and the
    flat region outside the bound simply never pays.
    """
    translation, rotation = params[:3].copy(), params[3:].copy()
    norm_t = float(np.linalg.norm(translation))
    if norm_t > max_translation:
        translation *= max_translation / norm_t
    norm_r = float(np.linalg.norm(rotation))
    if norm_r > max_rotation:
        rotation *= max_rotation / norm_r
    return translation, rotation


#: Pair separations beyond this contribute nothing worth computing to the
#: Lennard-Jones sum -- at 8 A the 6th-power term is already four orders below
#: its value at contact. A range cutoff, not a tuned weight.
_LJ_CUTOFF = 8.0
#: Below this the 12th power overflows; the objective is already astronomical
#: there, so clamping changes which minimum is found not at all.
_LJ_MIN_SEPARATION = 0.4
#: Penalty floor for a placement that overlaps more than stage one achieved.
#: Above any reachable Lennard-Jones value, and finite so Powell can compare it.
_INFEASIBLE = 1e12


class _Overlap:
    """Steric objective for a ligand against a fixed receptor.

    Two forms, and the difference between them is the whole point:

    ``"overlap"`` is the summed squared van der Waals overlap. It is zero the
    moment nothing overlaps, so its gradient dies there and the ligand stops
    pressed against the repulsive wall. Measured over 8,500 scored molecules,
    that is exactly where they end up: clash-free ones score -0.90 by Vina
    against the crystal ligands' -5.51, and 3.2 kcal/mol of that gap is
    recovered by a local optimisation that moves them only 1.36 A.

    ``"lj"`` is a Lennard-Jones 12-6 sum with sigma from the summed Bondi radii
    and unit well depth, which has a minimum at contact rather than a plateau
    past it -- the same rigid motion, allowed to settle into the well instead
    of stopping at its edge. Unit epsilon is what keeps it weightless: the
    radii set the only length scale and nothing trades off against anything.
    """

    def __init__(
        self,
        receptor_coords: np.ndarray,
        receptor_radii: np.ndarray,
        ligand_radii: np.ndarray,
        form: str = "overlap",
    ) -> None:
        self._tree = cKDTree(receptor_coords)
        self._coords = receptor_coords
        self._radii = receptor_radii
        self._ligand_radii = ligand_radii
        self._form = form
        self._clash_cutoff = CLASH_FRACTION * (
            float(ligand_radii.max()) + float(receptor_radii.max())
        )
        self._cutoff = _LJ_CUTOFF if form == "lj" else self._clash_cutoff

    def _neighbours(self, ligand_coords: np.ndarray, radius: float) -> list[list[int]]:
        return self._tree.query_ball_point(ligand_coords, r=radius)

    def __call__(self, ligand_coords: np.ndarray) -> float:
        total = 0.0
        for i, js in enumerate(self._neighbours(ligand_coords, self._cutoff)):
            if not js:
                continue
            idx = np.asarray(js)
            distance = np.linalg.norm(self._coords[idx] - ligand_coords[i], axis=1)
            if self._form == "lj":
                sigma = self._ligand_radii[i] + self._radii[idx]
                ratio = sigma / np.maximum(distance, _LJ_MIN_SEPARATION)
                six = ratio ** 6
                total += float(np.sum(six * six - 2.0 * six))
            else:
                gap = CLASH_FRACTION * (
                    self._ligand_radii[i] + self._radii[idx]
                ) - distance
                total += float(np.sum(np.maximum(gap, 0.0) ** 2))
        return total

    def clashing_atoms(self, ligand_coords: np.ndarray) -> int:
        count = 0
        for i, js in enumerate(self._neighbours(ligand_coords, self._clash_cutoff)):
            if not js:
                continue
            idx = np.asarray(js)
            distance = np.linalg.norm(self._coords[idx] - ligand_coords[i], axis=1)
            limit = CLASH_FRACTION * (self._ligand_radii[i] + self._radii[idx])
            if bool((distance < limit).any()):
                count += 1
        return count


@dataclass(frozen=True)
class RigidTransform:
    """A rotation about ``centre`` followed by a translation.

    Returned rather than applied so the caller can put the *same* transform on
    atoms the fit did not see -- hydrogens ride along with the heavy atoms they
    hang off. Recovering it from the moved coordinates instead would introduce
    a numerical step where an exact one is available.
    """

    centre: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray

    @property
    def shift(self) -> float:
        """Length of the translation, in Angstroms."""
        return float(np.linalg.norm(self.translation))

    def apply(self, coords: np.ndarray) -> np.ndarray:
        return (coords - self.centre) @ self.rotation.T + self.centre + self.translation


IDENTITY_AT_ORIGIN = RigidTransform(np.zeros(3), np.eye(3), np.zeros(3))


def rigid_pocket_fit(  # noqa: PLR0913
    ligand_coords: np.ndarray,
    ligand_radii: np.ndarray,
    receptor_coords: np.ndarray,
    receptor_radii: np.ndarray,
    *,
    max_translation: float = 2.5,
    max_rotation_deg: float = 30.0,
    n_restarts: int = 4,
    seed: int = 0,
    settle: bool = True,
) -> RigidTransform:
    """Return the bounded rigid transform that relieves the receptor overlap.

    The identity transform is always one of the candidates, so a pose that is
    already clash-free comes back untouched and the operation can only reduce
    the overlap it measures.
    """
    if len(receptor_coords) == 0 or len(ligand_coords) == 0:
        return IDENTITY_AT_ORIGIN

    overlap = _Overlap(receptor_coords, receptor_radii, ligand_radii)
    if not settle and overlap(ligand_coords) <= 0.0:
        # Stage one alone is at its minimum the moment nothing overlaps, so a
        # clash-free pose has nowhere to go. With settling there is always a
        # snugger placement, so the search runs either way.
        return IDENTITY_AT_ORIGIN

    centre = ligand_coords.mean(axis=0)
    centred = ligand_coords - centre
    max_rotation = np.deg2rad(max_rotation_deg)

    def place(params: np.ndarray) -> np.ndarray:
        translation, rotation = _clamp(params, max_translation, max_rotation)
        return centred @ _rotation(rotation).T + centre + translation

    def relieve(params: np.ndarray) -> float:
        return overlap(place(params))

    best_params = _search(relieve, n_restarts, seed)
    best_params = _shortest_equivalent(best_params, relieve, relieve(best_params))

    if settle:
        # Stage two: the overlap objective is flat once nothing overlaps, so
        # stage one leaves the ligand at the edge of the repulsive wall rather
        # than in the van der Waals well. Settling into the well is the same
        # rigid motion continued -- but never at the cost of the clash relief
        # just bought, so any placement that overlaps more than stage one's is
        # simply out of bounds rather than traded against.
        well = _Overlap(receptor_coords, receptor_radii, ligand_radii, "lj")
        ceiling = relieve(best_params) + _NEGLIGIBLE

        def snug(params: np.ndarray) -> float:
            coords = place(params)
            excess = overlap(coords) - ceiling
            if excess > 0.0:
                # A finite penalty rather than an infinity: Powell brackets by
                # comparing three points, and an inf among them makes the
                # comparison NaN, which walks the search off the map. The
                # offset is far above any reachable well depth, so infeasible
                # is always worse than feasible, and adding the excess back
                # gives the search a direction home.
                return _INFEASIBLE + excess
            return well(coords)

        best_params = _search(snug, n_restarts, seed, start=best_params)

    translation, rotation = _clamp(best_params, max_translation, max_rotation)
    return RigidTransform(centre, _rotation(rotation), translation)


def _search(
    objective: Callable[[np.ndarray], float],
    n_restarts: int,
    seed: int,
    start: np.ndarray | None = None,
) -> np.ndarray:
    """Powell from ``start`` plus random restarts; the best six parameters.

    ``start`` is always among the candidates, so the search can only improve on
    where it began. The restarts are there because Powell from one point stalls
    when the ligand is boxed in on several sides -- they give it a direction to
    fall in.
    """
    origin = np.zeros(6) if start is None else np.asarray(start, dtype=float)
    best_value, best_params = objective(origin), origin
    rng = np.random.default_rng(seed)
    # The first three parameters are a translation in Angstroms; everything
    # after is an angle in radians, whether it turns the whole molecule or one
    # of its bonds. Two scales, both of them restart spreads rather than model
    # parameters -- they say where to look, not what the answer costs.
    scale = np.concatenate(
        [np.full(3, 0.8), np.full(len(origin) - 3, 0.15)]
    )
    for _ in range(n_restarts):
        offset = rng.normal(0.0, 1.0, len(origin)) * scale
        result = minimize(
            objective, origin + offset, method="Powell",
            options={"maxiter": 400, "xtol": 0.05, "ftol": 0.05},
        )
        if result.fun < best_value:
            best_value, best_params = float(result.fun), result.x
    return best_params


def _shortest_equivalent(
    params: np.ndarray,
    objective: Callable[[np.ndarray], float],
    value: float,
) -> np.ndarray:
    """The smallest scaling of ``params`` whose objective still reaches ``value``.

    The overlap is not monotone along the path, so this scans rather than
    bisects; the identity end is included, which is what makes a pose that
    never needed moving come back unmoved.
    """
    tolerance = _NEGLIGIBLE + 1e-3 * abs(value)
    for fraction in np.linspace(0.0, 1.0, 21):
        scaled = params * fraction
        if objective(scaled) <= value + tolerance:
            return scaled
    return params

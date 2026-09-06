"""Pricing the rigid transform that a single-modality ligand tokenizer omits.

A tokenizer that encodes the ligand in its own canonical frame -- ConfSeq,
Token-Mol, Mol-StrucTok, and our own ``localframe_*`` arms -- produces an
SE(3)-invariant string. It says what the molecule looks like and nothing about
where in the receptor it sits, so putting the decoded molecule back costs a
rigid transform that has to be transmitted alongside. This module quantizes that
transform to a stated number of bits, so a row in a reconstruction table can be
read as "this fidelity, at this rate, plus this many bits of placement".

ProLIT's own arms spend **zero** bits here: pocket atoms and ligand atoms are
encoded as spherical coordinates in one shared pocket-canonical frame, so each
atom token already says where the atom is. That difference is the entire claim,
which is why it is priced rather than asserted.

Two surfaces, one quantizer:

* :func:`quantize_pose` returns the *values* -- what a reconstruction arm needs
  in order to place a decoded molecule and be scored on interface metrics.
* :func:`pose_code` / :func:`pose_tokens` return the same choice as an *integer*
  and as base-``2**token_bits`` digits -- what a language-model corpus needs in
  order to carry the placement as tokens the model predicts.

Both go through the same grids, so an LM trained on ``pose_bits`` tokens and a
reconstruction row measured at ``pose_bits`` sit on one rate-distortion curve.
Splitting them into two implementations is how two arms of one comparison end up
quantizing differently, which no test would catch.

This lives in ``prolit`` rather than beside either caller because both the
reconstruction benchmark and the corpus builder need it, and they are sibling
layers that must not import each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

__all__ = [
    "SemanticPoseCodec",
    "matrix_to_quat",
    "pose_code",
    "pose_from_code",
    "pose_tokens",
    "quantize_pose",
    "quat_to_matrix",
    "split_bits",
    "tokens_to_code",
    "translation_steps",
]

#: Bits carried by one token when the pose rides in an LM stream. Matches
#: ProLIT's 8192-entry codebook, so "one pose token" costs exactly what one
#: interface token costs and the budgets in a table are comparable.
DEFAULT_TOKEN_BITS = 13


def split_bits(pose_bits: int) -> tuple[int, int]:
    """Split a budget into (translation, rotation) bits.

    Half each, translation taking the floor. The split is fixed rather than
    tuned: tuning it per arm would let a baseline spend its budget where the
    metric happens to reward it, which is not a property of the representation.
    """
    trans_bits = pose_bits // 2
    return trans_bits, pose_bits - trans_bits


@lru_cache(maxsize=8)
def _rotation_grid(n_rot: int, seed: int = 0) -> np.ndarray:
    """``n_rot`` unit quaternions, deterministic in ``seed``.

    Cached: a 20-bit rotation budget is a million quaternions, and the corpus
    builder calls this once per ligand pose over tens of millions of them.
    """
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n_rot, 4))
    grid = q / np.linalg.norm(q, axis=1, keepdims=True)
    grid.setflags(write=False)
    return grid


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a unit quaternion ``(w, x, y, z)``."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(rot: np.ndarray) -> np.ndarray:
    """Unit quaternion ``(w, x, y, z)`` of a rotation matrix."""
    trace = np.trace(rot)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = np.array([
            0.25 / s,
            (rot[2, 1] - rot[1, 2]) * s,
            (rot[0, 2] - rot[2, 0]) * s,
            (rot[1, 0] - rot[0, 1]) * s,
        ])
    else:
        i = int(np.argmax(np.diag(rot)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + rot[i, i] - rot[j, j] - rot[k, k])
        q = np.zeros(4)
        q[0] = (rot[k, j] - rot[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (rot[j, i] + rot[i, j]) / s
        q[k + 1] = (rot[k, i] + rot[i, k]) / s
    return q / np.linalg.norm(q)


def translation_steps(trans_bits: int) -> int:
    """Divisions per axis affordable at ``trans_bits``: the largest ``s`` with
    ``s**3 <= 2**trans_bits``.

    The floor matters. Rounding ``2**(bits/3)`` to nearest, as the first
    implementation did, overspends whenever the cube root lands just above an
    integer -- at 19 bits it takes 81 divisions per axis, 531,441 cells against
    a 524,288-cell budget. Nothing breaks, and that is the problem: the arm is
    charged 39 bits in the table while spending 39.02, and the whole point of
    these rows is that the number of bits is the claim. The two budgets used
    before this was noticed (13 and 26 bits, i.e. 4 and 20 divisions) are
    unaffected; only 39 changes, 81 -> 80.
    """
    steps = max(int(2 ** (trans_bits / 3.0) + 1e-9), 1)
    while steps > 1 and steps**3 > 2**trans_bits:
        steps -= 1
    return steps


def _translation_cell(
    centroid: np.ndarray,
    box_origin: np.ndarray,
    box_size: float,
    trans_bits: int,
) -> tuple[np.ndarray, int]:
    """Cell index of ``centroid`` on the box's cubic grid, and its flat id."""
    steps = translation_steps(trans_bits)
    cell = box_size / steps
    idx = np.clip(np.floor((centroid - box_origin) / cell), 0, steps - 1)
    flat = int((idx[0] * steps + idx[1]) * steps + idx[2])
    return idx, flat


def quantize_pose(  # noqa: PLR0913
    centroid: np.ndarray,
    rotation: np.ndarray,
    box_origin: np.ndarray,
    box_size: float,
    pose_bits: int | None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a rigid transform to ``pose_bits`` bits.

    ``pose_bits=None`` returns it unchanged: oracle placement, the unattainable
    upper bound a rate argument is measured against.
    """
    if pose_bits is None:
        return centroid, rotation
    trans_bits, rot_bits = split_bits(pose_bits)
    steps = translation_steps(trans_bits)
    cell = box_size / steps
    idx, _ = _translation_cell(centroid, box_origin, box_size, trans_bits)
    centroid_q = box_origin + (idx + 0.5) * cell
    grid = _rotation_grid(2**rot_bits, seed)
    best = int(np.argmax(np.abs(grid @ matrix_to_quat(rotation))))
    return centroid_q, quat_to_matrix(grid[best])


def pose_code(  # noqa: PLR0913
    centroid: np.ndarray,
    rotation: np.ndarray,
    box_origin: np.ndarray,
    box_size: float,
    pose_bits: int,
    seed: int = 0,
) -> int:
    """The same quantization as :func:`quantize_pose`, as one integer.

    ``0 <= code < 2**pose_bits``: the translation cell in the high digits, the
    rotation grid index in the low ones.
    """
    trans_bits, rot_bits = split_bits(pose_bits)
    _, flat = _translation_cell(centroid, box_origin, box_size, trans_bits)
    grid = _rotation_grid(2**rot_bits, seed)
    rot_idx = int(np.argmax(np.abs(grid @ matrix_to_quat(rotation))))
    return flat * (2**rot_bits) + rot_idx


def pose_from_code(
    code: int,
    box_origin: np.ndarray,
    box_size: float,
    pose_bits: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`pose_code`: the centroid and rotation it stands for."""
    trans_bits, rot_bits = split_bits(pose_bits)
    steps = translation_steps(trans_bits)
    flat, rot_idx = divmod(int(code), 2**rot_bits)
    idx = np.array([flat // (steps * steps), (flat // steps) % steps, flat % steps])
    centroid = box_origin + (idx + 0.5) * (box_size / steps)
    return centroid, quat_to_matrix(_rotation_grid(2**rot_bits, seed)[rot_idx])


def pose_tokens(
    code: int,
    pose_bits: int,
    token_bits: int = DEFAULT_TOKEN_BITS,
) -> tuple[int, ...]:
    """``code`` as fixed-length base-``2**token_bits`` digits, most significant
    first.

    Fixed length, not minimal: a variable number of pose tokens would make the
    ligand block's length depend on the value, and the model would read the
    length as a hint about the answer.
    """
    n = -(-pose_bits // token_bits)
    base = 2**token_bits
    digits = []
    for _ in range(n):
        code, rem = divmod(int(code), base)
        digits.append(rem)
    return tuple(reversed(digits))


def tokens_to_code(
    tokens: tuple[int, ...],
    token_bits: int = DEFAULT_TOKEN_BITS,
) -> int:
    """Inverse of :func:`pose_tokens`."""
    code = 0
    for t in tokens:
        code = code * (2**token_bits) + int(t)
    return code


# ---------------------------------------------------------------------------
# The language-model surface: the same budget, spent so a model can learn it.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticPoseCodec:
    """Four tokens that mean *where roughly*, *where exactly*, *how roughly*,
    *how exactly*.

    :func:`pose_code` packs the whole transform into one integer, which is right
    for a reconstruction table -- the number is only ever decoded, never
    predicted. Handing that integer to a language model as base-8192 digits is
    not: the digit boundaries fall in the middle of the translation raster and
    again in the middle of the rotation index, so two poses 0.3 A apart get
    unrelated token ids. The code is a hash, and a model can only memorise it.

    Here each token is one quantity at one scale, and each refines the one
    before it, so poses that agree to within a coarse cell share their leading
    tokens and differ only in the trailing ones.

    **Why four and not three.** Three tokens is 39 bits, matching the
    reconstruction sweep. Measured over 400 random placements in a 30 A box,
    every three-token split leaves the pose channel coarser than the effects
    being measured: translation over one token quantizes to 1.5 A cells
    (0.75 A median error), while spending two tokens on translation leaves
    rotation at 13 bits and a 16 deg worst case, which is 0.8 A at the rim of a
    drug-sized ligand. Docking power is decided in the 0-1 A band, so either
    split would make **this quantizer**, not the representation, the thing the
    experiment measures. Four tokens put the placement error at roughly 0.04 A,
    an order of magnitude below anything the tasks resolve, which takes the
    choice out of the result. The extra 13 bits are charged to the baseline in
    the rate column and stated in the paper; they are spent in the baseline's
    favour, and the reconstruction table already shows ProLIT ahead of this
    construction even at *oracle* placement.
    """

    #: Divisions per axis, at each of the two translation scales. 20**3 = 8000
    #: fits one 13-bit token; two levels reach 400 divisions per axis.
    trans_cells: int = 20
    coarse: int = 8192
    fine: int = 8192
    #: Angular reach of the refinement grid, set from what the coarse grid
    #: actually leaves behind: over 20k random rotations its nearest neighbour
    #: is 4.17 deg away at the median, 14.07 at the 99.9th percentile and 16.05
    #: at the worst. 18 deg covers that with margin. Wider would spend the fine
    #: token's resolution on angles the coarse token never leaves; narrower
    #: would leave a gap the refinement cannot cross.
    fine_max_deg: float = 18.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.trans_cells**3 > 2**DEFAULT_TOKEN_BITS:
            msg = (
                f"{self.trans_cells}**3 cells does not fit in one "
                f"{DEFAULT_TOKEN_BITS}-bit token"
            )
            raise ValueError(msg)

    @property
    def n_tokens(self) -> int:
        return 4

    @property
    def bits(self) -> float:
        """What this actually spends, for the rate column."""
        return float(
             2 * np.log2(self.trans_cells**3)
            + np.log2(self.coarse)
            + np.log2(self.fine)
        )

    def _cell(self, value: np.ndarray, origin: np.ndarray, size: float) -> tuple:
        step = size / self.trans_cells
        idx = np.clip(np.floor((value - origin) / step), 0, self.trans_cells - 1)
        flat = int((idx[0] * self.trans_cells + idx[1]) * self.trans_cells + idx[2])
        return idx, flat, step

    def encode(
        self,
        centroid: np.ndarray,
        rotation: np.ndarray,
        box_origin: np.ndarray,
        box_size: float,
    ) -> tuple[int, int, int, int]:
        """(translation coarse, translation fine, rotation coarse, rotation fine)."""
        idx, t_coarse, step = self._cell(centroid, box_origin, box_size)
        sub_origin = box_origin + idx * step
        _, t_fine, _ = self._cell(centroid, sub_origin, step)

        q = matrix_to_quat(rotation)
        g1 = _rotation_grid(self.coarse, self.seed)
        r_coarse = int(np.argmax(np.abs(g1 @ q)))
        # What the coarse token leaves undone, expressed as a rotation.
        residual = rotation @ quat_to_matrix(g1[r_coarse]).T
        g2 = _refinement_grid(self.fine, self.fine_max_deg, self.seed)
        r_fine = int(np.argmax(np.abs(g2 @ matrix_to_quat(residual))))
        return t_coarse, t_fine, r_coarse, r_fine

    def decode(
        self,
        tokens: tuple[int, ...],
        box_origin: np.ndarray,
        box_size: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inverse of :meth:`encode`."""
        t_coarse, t_fine, r_coarse, r_fine = (int(v) for v in tokens)
        n = self.trans_cells
        step = box_size / n
        idx = np.array([t_coarse // (n * n), (t_coarse // n) % n, t_coarse % n])
        sub = np.array([t_fine // (n * n), (t_fine // n) % n, t_fine % n])
        centroid = box_origin + idx * step + (sub + 0.5) * (step / n)

        coarse = quat_to_matrix(_rotation_grid(self.coarse, self.seed)[r_coarse])
        fine = quat_to_matrix(
            _refinement_grid(self.fine, self.fine_max_deg, self.seed)[r_fine]
        )
        return centroid, fine @ coarse


@lru_cache(maxsize=4)
def _refinement_grid(n: int, max_deg: float, seed: int = 0) -> np.ndarray:
    """``n`` small rotations, as quaternions, out to ``max_deg``.

    Axis uniform on the sphere; angle drawn so the rotations are spread evenly
    through the cap rather than piled at its centre. Haar measure on SO(3) puts
    density proportional to ``1 - cos(theta)``, which is ``theta**2 / 2`` to
    within 1% over the 30 degrees this covers, so the inverse CDF is a cube
    root.
    """
    rng = np.random.default_rng(seed + 1_000_003)
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    theta = np.deg2rad(max_deg) * rng.random(n) ** (1.0 / 3.0)
    half = theta / 2.0
    grid = np.concatenate([np.cos(half)[:, None], axis * np.sin(half)[:, None]], axis=1)
    grid.setflags(write=False)
    return grid

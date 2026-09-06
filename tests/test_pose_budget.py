"""The pose quantizer is shared, so the two callers must not drift apart.

``quantize_pose`` prices the rigid transform a ligand-own-frame tokenizer omits.
Two places need it -- the reconstruction benchmark, which places a decoded
molecule to score interface metrics, and the corpus builder, which writes the
placement into an LM stream as tokens. If those quantize differently, the
reconstruction row and the language-model row of the same arm are measured on
different curves while both are labelled with the same number of bits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from prolit.tokenizers.pose_budget import (
    matrix_to_quat,
    pose_code,
    pose_from_code,
    pose_tokens,
    quantize_pose,
    quat_to_matrix,
    split_bits,
    tokens_to_code,
    translation_steps,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

BOX_ORIGIN = np.array([-10.0, -8.0, -12.0])
BOX_SIZE = 24.0


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    return quat_to_matrix(q / np.linalg.norm(q))


def _cases(n: int = 32) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(11)
    for _ in range(n):
        centroid = BOX_ORIGIN + rng.uniform(0.0, BOX_SIZE, size=3)
        yield centroid, _random_rotation(rng)


@pytest.mark.parametrize("pose_bits", [13, 26, 39])
def test_code_round_trip_matches_quantize(pose_bits: int) -> None:
    """The integer form and the value form are the same quantization."""
    for centroid, rotation in _cases():
        cq, rq = quantize_pose(centroid, rotation, BOX_ORIGIN, BOX_SIZE, pose_bits)
        code = pose_code(centroid, rotation, BOX_ORIGIN, BOX_SIZE, pose_bits)
        cd, rd = pose_from_code(code, BOX_ORIGIN, BOX_SIZE, pose_bits)
        assert np.allclose(cq, cd)
        # A quaternion and its negation are the same rotation; compare matrices.
        assert np.allclose(rq, rd)


@pytest.mark.parametrize("pose_bits", [13, 26, 39])
def test_code_fits_its_budget(pose_bits: int) -> None:
    for centroid, rotation in _cases():
        code = pose_code(centroid, rotation, BOX_ORIGIN, BOX_SIZE, pose_bits)
        assert 0 <= code < 2**pose_bits


@pytest.mark.parametrize("pose_bits", [13, 26, 39])
def test_tokens_round_trip(pose_bits: int) -> None:
    """Fixed-length digits, so the ligand block's length never leaks the value."""
    n_expected = -(-pose_bits // 13)
    for centroid, rotation in _cases():
        code = pose_code(centroid, rotation, BOX_ORIGIN, BOX_SIZE, pose_bits)
        toks = pose_tokens(code, pose_bits)
        assert len(toks) == n_expected
        assert all(0 <= t < 8192 for t in toks)
        assert tokens_to_code(toks) == code


def test_oracle_is_the_identity() -> None:
    """``pose_bits=None`` is the unattainable upper bound, not a fine grid."""
    for centroid, rotation in _cases(4):
        cq, rq = quantize_pose(centroid, rotation, BOX_ORIGIN, BOX_SIZE, None)
        assert cq is centroid
        assert rq is rotation


def test_more_bits_never_hurts() -> None:
    """Distortion is monotone in the budget -- the property a rate curve needs."""
    errs = {}
    for bits in (13, 26, 39):
        trans, _ = split_bits(bits)
        errs[bits] = BOX_SIZE / translation_steps(trans)
    assert errs[13] > errs[26] > errs[39]
    # And the rotation half actually gets finer too.
    rot_err = {}
    for bits in (13, 26, 39):
        worst = 0.0
        for _, rotation in _cases(8):
            _, rq = quantize_pose(
                np.zeros(3), rotation, BOX_ORIGIN, BOX_SIZE, bits, seed=0
            )
            dot = abs(float(matrix_to_quat(rotation) @ matrix_to_quat(rq)))
            worst = max(worst, 2 * np.arccos(min(dot, 1.0)))
        rot_err[bits] = worst
    assert rot_err[39] < rot_err[13]


def test_the_budget_is_not_overspent() -> None:
    """The cell grid fits inside the bits it is charged for.

    The first implementation rounded the cube root to nearest and took 81
    divisions per axis at 19 translation bits -- 531,441 cells priced as
    524,288. A table whose whole content is "this many bits" cannot afford
    that, and it is invisible unless something counts.
    """
    for bits in (13, 20, 26, 33, 39, 52):
        trans, _ = split_bits(bits)
        assert translation_steps(trans) ** 3 <= 2**trans


def test_matches_the_reconstruction_cli() -> None:
    """Bit-identical to the implementation the published localframe rows used.

    The CLI now imports this module; this pins the numbers it produced before
    it did, so moving the function did not silently move the curve.
    """
    rng = np.random.default_rng(3)

    def _grid(n_rot: int, seed: int = 0) -> np.ndarray:
        r = np.random.default_rng(seed)
        q = r.normal(size=(n_rot, 4))
        return q / np.linalg.norm(q, axis=1, keepdims=True)

    def _reference(  # noqa: PLR0913
        centroid: np.ndarray,
        rotation: np.ndarray,
        box_origin: np.ndarray,
        box_size: float,
        pose_bits: int,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        if pose_bits is None:
            return centroid, rotation
        trans_bits = pose_bits // 2
        rot_bits = pose_bits - trans_bits
        steps = max(round(2 ** (trans_bits / 3.0)), 1)
        cell = box_size / steps
        idx = np.clip(np.floor((centroid - box_origin) / cell), 0, steps - 1)
        centroid_q = box_origin + (idx + 0.5) * cell
        grid = _grid(2**rot_bits, seed)
        best = int(np.argmax(np.abs(grid @ matrix_to_quat(rotation))))
        return centroid_q, quat_to_matrix(grid[best])

    for pose_bits in (13, 26):
        for _ in range(16):
            centroid = BOX_ORIGIN + rng.uniform(0.0, BOX_SIZE, size=3)
            rotation = _random_rotation(rng)
            got = quantize_pose(centroid, rotation, BOX_ORIGIN, BOX_SIZE, pose_bits)
            want = _reference(centroid, rotation, BOX_ORIGIN, BOX_SIZE, pose_bits)
            assert np.allclose(got[0], want[0])
            assert np.allclose(got[1], want[1])

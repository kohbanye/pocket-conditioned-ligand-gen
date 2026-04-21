"""Shared geometry utilities for internal-coordinate tokenizers.

Provides:
- Bond angle / dihedral angle computation
- NeRF atom placement
- Spherical coordinate helpers
- Virtual reference point construction
- Sinusoidal positional encoding
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def bond_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the angle (radians) at *b* in the triangle a-b-c."""
    v1 = a - b
    v2 = c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < _EPS:
        return 0.0
    cos_angle = np.dot(v1, v2) / denom
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def dihedral_angle(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> float:
    """Dihedral angle (radians) defined by four points p0-p1-p2-p3."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < _EPS or n2_norm < _EPS:
        return 0.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    b2_unit = b2 / (np.linalg.norm(b2) + _EPS)
    m = np.cross(n1, b2_unit)

    return float(np.arctan2(np.dot(m, n2), np.dot(n1, n2)))


def place_atom(  # noqa: PLR0913
    ref_a: np.ndarray,
    ref_b: np.ndarray,
    ref_c: np.ndarray,
    bond_length: float,
    angle: float,
    torsion: float,
) -> np.ndarray:
    """Place atom D via NeRF so that ``dihedral(A, B, C, D) == torsion``.

    *ref_a*, *ref_b*, *ref_c* correspond to dihedral_ref, angle_ref, parent.
    """
    v1 = ref_b - ref_c
    v1_norm = np.linalg.norm(v1)
    if v1_norm < _EPS:
        return ref_c + np.array([bond_length, 0.0, 0.0])
    v1_hat = v1 / v1_norm

    v2 = ref_a - ref_b
    n = np.cross(v1_hat, v2)
    n_norm = np.linalg.norm(n)

    if n_norm < _EPS:
        perp = (
            np.array([1.0, 0.0, 0.0])
            if abs(v1_hat[0]) < 0.9  # noqa: PLR2004
            else np.array([0.0, 1.0, 0.0])
        )
        n = np.cross(v1_hat, perp)
        n = n / np.linalg.norm(n)
    else:
        n = n / n_norm

    m = np.cross(n, v1_hat)

    return ref_c + bond_length * (
        np.cos(angle) * v1_hat
        + np.sin(angle) * np.cos(torsion) * m
        + np.sin(angle) * np.sin(torsion) * n
    )


def virtual_dihedral_ref(
    angle_ref_pos: np.ndarray,
    parent_pos: np.ndarray,
) -> np.ndarray:
    """Construct a deterministic virtual dihedral reference point.

    Used when no real dihedral reference is available (DFS positions 0-2 and
    disconnected-component starts).  The virtual point is placed so that
    ``torsion = 0`` produces a deterministic, reproducible placement.
    """
    v = angle_ref_pos - parent_pos
    v_hat = v / (np.linalg.norm(v) + _EPS)
    up = (
        np.array([0.0, 1.0, 0.0])
        if abs(v_hat[1]) < 0.9  # noqa: PLR2004
        else np.array([0.0, 0.0, 1.0])
    )
    perp = up - np.dot(up, v_hat) * v_hat
    perp = perp / (np.linalg.norm(perp) + _EPS)
    return angle_ref_pos + perp


def canonical_virtual_ref(
    angle_ref_pos: np.ndarray,
    parent_pos: np.ndarray,
) -> np.ndarray:
    """Virtual dihedral ref aligned with the canonical frame z-axis.

    Used in pocket-anchored mode so that the torsion encodes
    orientation relative to the pocket's canonical frame.
    Falls back to x-axis if the bond is nearly parallel to z.
    """
    bond = parent_pos - angle_ref_pos
    bond_norm = np.linalg.norm(bond)
    if bond_norm < _EPS:
        return angle_ref_pos + np.array([0.0, 0.0, 1.0])
    bond_hat = bond / bond_norm
    axis = (
        np.array([1.0, 0.0, 0.0])
        if abs(bond_hat[2]) > 0.9  # noqa: PLR2004
        else np.array([0.0, 0.0, 1.0])
    )
    return angle_ref_pos + axis


# ---------------------------------------------------------------------------
# Spherical coordinate helpers (for pocket-anchored mode)
# ---------------------------------------------------------------------------


def cartesian_to_spherical(
    pos: np.ndarray,
) -> tuple[float, float, float, float]:
    """Convert 3D Cartesian to ``(r, theta_polar, sin phi, cos phi)``."""
    r = float(np.linalg.norm(pos))
    if r < _EPS:
        return 0.0, 0.0, 0.0, 1.0
    theta = float(np.arccos(np.clip(pos[2] / r, -1.0, 1.0)))
    phi = float(np.arctan2(pos[1], pos[0]))
    return r, theta, float(np.sin(phi)), float(np.cos(phi))


def spherical_to_cartesian(
    r: float,
    theta: float,
    sin_phi: float,
    cos_phi: float,
) -> np.ndarray:
    """Convert ``(r, theta_polar, sin phi, cos phi)`` to 3D Cartesian."""
    phi = np.arctan2(sin_phi, cos_phi)
    return np.array(
        [
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ]
    )


# ---------------------------------------------------------------------------
# Batched differentiable torch ports (used by the coord-reconstruction loss)
# ---------------------------------------------------------------------------


def spherical_to_cartesian_batched(
    r: Tensor,
    theta: Tensor,
    sin_phi: Tensor,
    cos_phi: Tensor,
) -> Tensor:
    """Batched ``(r, theta, sin phi, cos phi)`` → Cartesian ``(..., 3)``.

    Inputs broadcast over the leading dims; caller is responsible for
    projecting ``(sin_phi, cos_phi)`` onto the unit circle.
    """
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    x = r * sin_theta * cos_phi
    y = r * sin_theta * sin_phi
    z = r * cos_theta
    return torch.stack([x, y, z], dim=-1)


def canonical_virtual_ref_batched(
    angle_ref_pos: Tensor,
    parent_pos: Tensor,
) -> Tensor:
    """Batched canonical virtual dihedral reference (``(..., 3)`` → ``(..., 3)``).

    Matches :func:`canonical_virtual_ref`: selects an axis not parallel to
    the parent→angle-ref bond direction. Zero-length bond falls back to
    the z-axis offset (same as the numpy branch).
    """
    bond = parent_pos - angle_ref_pos
    bond_norm_sq = (bond * bond).sum(dim=-1, keepdim=True)
    bond_norm = bond_norm_sq.clamp_min(_EPS * _EPS).sqrt()
    bond_hat = bond / bond_norm

    z_axis = angle_ref_pos.new_tensor([0.0, 0.0, 1.0])
    x_axis = angle_ref_pos.new_tensor([1.0, 0.0, 0.0])
    use_x = bond_hat[..., 2:3].abs() > 0.9  # noqa: PLR2004
    axis = torch.where(use_x, x_axis, z_axis)
    return angle_ref_pos + axis


def place_atom_batched(  # noqa: PLR0913
    ref_a: Tensor,
    ref_b: Tensor,
    ref_c: Tensor,
    bond_length: Tensor,
    angle: Tensor,
    sin_tau: Tensor,
    cos_tau: Tensor,
) -> Tensor:
    """Batched differentiable NeRF placement.

    ``ref_a``, ``ref_b``, ``ref_c`` are ``(..., 3)`` positions
    (dihedral_ref, angle_ref, parent). Scalars ``bond_length``, ``angle``,
    ``sin_tau``, ``cos_tau`` have shape ``(...)``. Caller projects
    ``(sin_tau, cos_tau)`` onto the unit circle. Matches
    :func:`place_atom` numerically.
    """
    v1 = ref_b - ref_c
    v1_norm_sq = (v1 * v1).sum(dim=-1, keepdim=True)
    v1_norm = v1_norm_sq.clamp_min(_EPS * _EPS).sqrt()
    v1_hat = v1 / v1_norm

    v2 = ref_a - ref_b
    n = torch.linalg.cross(v1_hat, v2, dim=-1)
    n_norm_sq = (n * n).sum(dim=-1, keepdim=True)

    # Degenerate-cross-product fallback: pick a perpendicular axis
    # that is not parallel to v1_hat.
    x_axis = v1.new_tensor([1.0, 0.0, 0.0])
    y_axis = v1.new_tensor([0.0, 1.0, 0.0])
    use_x = v1_hat[..., 0:1].abs() < 0.9  # noqa: PLR2004
    perp = torch.where(use_x, x_axis, y_axis)
    n_fallback = torch.linalg.cross(v1_hat, perp, dim=-1)

    use_fallback = n_norm_sq < (_EPS * _EPS)
    n_selected = torch.where(use_fallback, n_fallback, n)
    n_sel_norm_sq = (n_selected * n_selected).sum(dim=-1, keepdim=True)
    n_hat = n_selected / n_sel_norm_sq.clamp_min(_EPS * _EPS).sqrt()

    m = torch.linalg.cross(n_hat, v1_hat, dim=-1)

    cos_ang = torch.cos(angle).unsqueeze(-1)
    sin_ang = torch.sin(angle).unsqueeze(-1)
    s_tau = sin_tau.unsqueeze(-1)
    c_tau = cos_tau.unsqueeze(-1)
    bl = bond_length.unsqueeze(-1)

    return ref_c + bl * (
        cos_ang * v1_hat + sin_ang * c_tau * m + sin_ang * s_tau * n_hat
    )


def project_unit_circle(sin_v: Tensor, cos_v: Tensor) -> tuple[Tensor, Tensor]:
    """Project ``(sin, cos)`` onto the unit circle to stabilise NeRF inputs.

    Network outputs are not constrained to ``sin² + cos² = 1``; feeding
    unnormalised values into :func:`place_atom_batched` produces
    out-of-scale placements. Safe under autograd.
    """
    norm_sq = sin_v * sin_v + cos_v * cos_v
    norm = norm_sq.clamp_min(_EPS * _EPS).sqrt()
    return sin_v / norm, cos_v / norm


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> Tensor:
    """Create sinusoidal positional encoding table."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

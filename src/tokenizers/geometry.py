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

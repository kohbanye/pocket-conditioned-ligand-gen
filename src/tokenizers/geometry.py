"""Shared geometry utilities for the spherical-from-pocket-centroid tokenizer.

After the move away from Z-matrix dihedral chains, we only need:
- Cartesian ↔ spherical conversion (numpy + batched torch).
- Unit-circle projection for the (sin φ, cos φ) decoder slots.
- Sinusoidal positional encoding for the Transformer.

NeRF placement, virtual dihedral references, and bond/dihedral-angle helpers
were removed when the descriptor format changed; consult the git history if
you need to recover them.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Spherical coordinate helpers
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


def spherical_to_cartesian_batched(
    r: Tensor,
    theta: Tensor,
    sin_phi: Tensor,
    cos_phi: Tensor,
) -> Tensor:
    """Batched spherical → Cartesian. Caller projects (sin_phi, cos_phi) first."""
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    x = r * sin_theta * cos_phi
    y = r * sin_theta * sin_phi
    z = r * cos_theta
    return torch.stack([x, y, z], dim=-1)


def project_unit_circle(sin_v: Tensor, cos_v: Tensor) -> tuple[Tensor, Tensor]:
    """Project ``(sin, cos)`` onto the unit circle.

    Network outputs are not constrained to ``sin² + cos² = 1``; feeding
    unnormalised values into spherical → Cartesian produces out-of-scale
    placements. Safe under autograd (clamp avoids the ``r=0`` singularity).
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

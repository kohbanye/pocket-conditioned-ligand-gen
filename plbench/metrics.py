"""Reconstruction-quality metrics.

All functions take two (N, 3) arrays with a known 1:1 row correspondence
(reference, reconstructed) and return scalars. Pure NumPy, no model deps.
"""

from __future__ import annotations

import numpy as np


def _as2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 3:  # (L, atoms_per_res, 3) -> (L*atoms, 3)
        x = x.reshape(-1, 3)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"expected (N, 3) coordinates, got shape {x.shape}")
    return x


def kabsch_rotation(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Optimal rotation mapping centered ``p`` onto centered ``q`` (proper)."""
    h = p.T @ q
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    e = np.diag([1.0, 1.0, d])
    return u @ e @ vt


def superpose(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return ``p`` rigidly superposed onto ``q`` (Kabsch)."""
    p, q = _as2d(p), _as2d(q)
    pc, qc = p.mean(0), q.mean(0)
    r = kabsch_rotation(p - pc, q - qc)
    return (p - pc) @ r + qc


def rmsd(p: np.ndarray, q: np.ndarray) -> float:
    """RMSD without superposition (coordinates compared as given)."""
    p, q = _as2d(p), _as2d(q)
    return float(np.sqrt(np.mean(np.sum((p - q) ** 2, axis=1))))


def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    """RMSD after optimal rigid superposition of ``p`` onto ``q``."""
    return rmsd(superpose(p, q), q)


def tm_score(ref: np.ndarray, rec: np.ndarray) -> float:
    """TM-score for a known 1:1 CA correspondence (superpose, then TM formula).

    Normalized by the reference length L, d0 = 1.24*(L-15)^(1/3) - 1.8.
    """
    ref, rec = _as2d(ref), _as2d(rec)
    n = ref.shape[0]
    if n == 0:
        return float("nan")
    aligned = superpose(rec, ref)
    d2 = np.sum((aligned - ref) ** 2, axis=1)
    if n > 15:
        d0 = 1.24 * (n - 15) ** (1.0 / 3.0) - 1.8
    else:
        d0 = 0.5
    d0 = max(d0, 0.5)
    return float(np.mean(1.0 / (1.0 + d2 / (d0 * d0))))


def lddt(
    ref: np.ndarray,
    rec: np.ndarray,
    cutoff: float = 15.0,
    thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> float:
    """Superposition-free lDDT on CA atoms (Mariani et al. 2013).

    Fraction of reference inter-residue distances (< ``cutoff``) that are
    preserved within the distance ``thresholds`` in the reconstruction.
    """
    ref, rec = _as2d(ref), _as2d(rec)
    n = ref.shape[0]
    if n < 2:
        return float("nan")
    dref = np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=-1)
    drec = np.linalg.norm(rec[:, None, :] - rec[None, :, :], axis=-1)
    mask = (dref < cutoff) & ~np.eye(n, dtype=bool)
    if not mask.any():
        return float("nan")
    diff = np.abs(dref - drec)[mask]
    preserved = np.mean([(diff < t).mean() for t in thresholds])
    return float(preserved)


def all_metrics(ref: np.ndarray, rec: np.ndarray, *, protein: bool) -> dict[str, float]:
    """Compute the standard metric set for one aligned (ref, rec) pair."""
    out: dict[str, float] = {
        "rmsd": rmsd(ref, rec),
        "kabsch_rmsd": kabsch_rmsd(ref, rec),
        "n_atoms": int(_as2d(ref).shape[0]),
    }
    if protein:
        out["tm_score"] = tm_score(ref, rec)
        out["lddt"] = lddt(ref, rec)
    return out

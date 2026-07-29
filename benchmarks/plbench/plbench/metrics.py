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


# ---------------------------------------------------------------------------
# Interface metrics
#
# A tokenizer can reconstruct a pocket and a ligand each accurately and still
# lose the thing that matters for binding: where the ligand sits relative to the
# receptor. Superposing each modality separately hides that, so these functions
# score the two together, in the frame they were reconstructed in.
# ---------------------------------------------------------------------------

_VDW_FALLBACK = 1.7
_VDW: dict[str, float] = {}


def vdw_radius(symbol: str) -> float:
    """van der Waals radius in Angstrom (RDKit periodic table), carbon fallback."""
    if symbol not in _VDW:
        try:
            from rdkit.Chem import GetPeriodicTable

            _VDW[symbol] = float(GetPeriodicTable().GetRvdw(symbol))
        except (ImportError, RuntimeError):
            _VDW[symbol] = _VDW_FALLBACK
    return _VDW[symbol]


def lddt_pli(
    ref_protein: np.ndarray,
    ref_ligand: np.ndarray,
    rec_protein: np.ndarray,
    rec_ligand: np.ndarray,
    inclusion_radius: float = 6.0,
    thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> float:
    """CASP15-style lDDT-PLI: lDDT restricted to protein-ligand atom pairs.

    Superposition-free, so it measures whether the *relative* placement survived
    tokenization. Reference pairs closer than ``inclusion_radius`` are scored.
    """
    ref_p, ref_l = _as2d(ref_protein), _as2d(ref_ligand)
    rec_p, rec_l = _as2d(rec_protein), _as2d(rec_ligand)
    dref = np.linalg.norm(ref_p[:, None, :] - ref_l[None, :, :], axis=-1)
    drec = np.linalg.norm(rec_p[:, None, :] - rec_l[None, :, :], axis=-1)
    mask = dref < inclusion_radius
    if not mask.any():
        return float("nan")
    diff = np.abs(dref - drec)[mask]
    return float(np.mean([(diff < t).mean() for t in thresholds]))


def contact_prf(
    ref_protein: np.ndarray,
    ref_ligand: np.ndarray,
    rec_protein: np.ndarray,
    rec_ligand: np.ndarray,
    cutoff: float = 4.0,
) -> tuple[float, float, float]:
    """Precision / recall / F1 of the protein-ligand contact set at ``cutoff``."""
    ref_c = (
        np.linalg.norm(_as2d(ref_protein)[:, None, :] - _as2d(ref_ligand)[None, :, :], axis=-1)
        < cutoff
    )
    rec_c = (
        np.linalg.norm(_as2d(rec_protein)[:, None, :] - _as2d(rec_ligand)[None, :, :], axis=-1)
        < cutoff
    )
    tp = float((ref_c & rec_c).sum())
    precision = tp / rec_c.sum() if rec_c.sum() else float("nan")
    recall = tp / ref_c.sum() if ref_c.sum() else float("nan")
    if not (precision > 0 and recall > 0):
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def clash_stats(
    protein: np.ndarray,
    ligand: np.ndarray,
    protein_elements: list[str],
    ligand_elements: list[str],
    tolerance: float = 0.75,
) -> dict[str, float]:
    """Steric clashes between the reconstructed protein and ligand.

    A pair clashes when its distance falls below ``tolerance`` times the sum of
    van der Waals radii -- the PoseBusters convention.
    """
    p, lig = _as2d(protein), _as2d(ligand)
    p_vdw = np.array([vdw_radius(e) for e in protein_elements])
    l_vdw = np.array([vdw_radius(e) for e in ligand_elements])
    dist = np.linalg.norm(p[:, None, :] - lig[None, :, :], axis=-1)
    limit = tolerance * (p_vdw[:, None] + l_vdw[None, :])
    return {
        "clash_pair_frac": float((dist < limit).mean()),
        "clash_lig_atom_frac": float((dist < limit).any(axis=0).mean()),
        "min_dist_ratio": float((dist / (p_vdw[:, None] + l_vdw[None, :])).min()),
    }


def bond_geometry(
    ref: np.ndarray, rec: np.ndarray, bonds: list[tuple[int, int]]
) -> dict[str, float]:
    """Bond-length and bond-angle absolute errors, mean and worst-case.

    The worst-case columns matter: a tokenizer can look fine on average while
    mangling one bond per molecule, which is what breaks chemical validity.
    """
    if not bonds:
        return dict.fromkeys(("bond_mae", "bond_max", "angle_mae", "angle_max"), np.nan)
    ref, rec = _as2d(ref), _as2d(rec)
    i = np.array([b[0] for b in bonds])
    j = np.array([b[1] for b in bonds])
    len_err = np.abs(
        np.linalg.norm(rec[i] - rec[j], axis=-1) - np.linalg.norm(ref[i] - ref[j], axis=-1)
    )

    neighbors: dict[int, list[int]] = {}
    for a, b in bonds:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)

    def angles(coords: np.ndarray) -> np.ndarray:
        out = []
        for center, nbrs in neighbors.items():
            for a_i in range(len(nbrs)):
                for b_i in range(a_i + 1, len(nbrs)):
                    v1 = coords[nbrs[a_i]] - coords[center]
                    v2 = coords[nbrs[b_i]] - coords[center]
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    # Coincident atoms still take a slot so the reference and
                    # model lists stay index-aligned; a tokenizer that collapses
                    # two atoms onto one point would otherwise shift every
                    # later angle and corrupt the whole comparison.
                    out.append(
                        np.nan
                        if n1 < 1e-6 or n2 < 1e-6
                        else np.degrees(
                            np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                        )
                    )
        return np.array(out)

    ref_ang, rec_ang = angles(ref), angles(rec)
    ang_err = np.abs(rec_ang - ref_ang) if ref_ang.size else np.array([np.nan])
    finite = ang_err[np.isfinite(ang_err)]
    return {
        "bond_mae": float(len_err.mean()),
        "bond_max": float(len_err.max()),
        "angle_mae": float(finite.mean()) if finite.size else np.nan,
        "angle_max": float(finite.max()) if finite.size else np.nan,
    }


def complex_metrics(
    ref: np.ndarray,
    rec: np.ndarray,
    n_protein: int,
    protein_elements: list[str],
    ligand_elements: list[str],
) -> dict[str, float]:
    """Interface metrics for a stacked ``[protein; ligand]`` reconstruction."""
    ref, rec = _as2d(ref), _as2d(rec)
    ref_p, ref_l = ref[:n_protein], ref[n_protein:]
    rec_p, rec_l = rec[:n_protein], rec[n_protein:]
    if ref_l.shape[0] == 0 or ref_p.shape[0] == 0:
        return {}
    precision, recall, f1 = contact_prf(ref_p, ref_l, rec_p, rec_l)
    # RMSD over just the ligand atoms that actually touch the receptor.
    iface = (np.linalg.norm(ref_p[:, None, :] - ref_l[None, :, :], axis=-1) < 4.0).any(axis=0)
    return {
        "lddt_pli": lddt_pli(ref_p, ref_l, rec_p, rec_l),
        "contact_precision": precision,
        "contact_recall": recall,
        "contact_f1": f1,
        "iface_lig_rmsd": rmsd(ref_l[iface], rec_l[iface]) if iface.any() else np.nan,
        **clash_stats(rec_p, rec_l, protein_elements, ligand_elements),
    }

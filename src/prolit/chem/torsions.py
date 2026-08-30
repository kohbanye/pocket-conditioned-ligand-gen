"""Rotatable bonds and the atoms each one moves.

A refiner that emits torsion angles cannot change a bond length or a bond
angle, because a rotation about a bond axis preserves every distance except
those spanning the bond. That is the whole point of the output space: measured
over 60 targets, every free-displacement refiner in this repository buys
contact by shrinking the molecule (bonds out of tolerance 10.0% without one,
48.1% with ``refit_press0.6``), and weighting the bond loss cannot stop it --
bonded pairs carry only 0.077 of the reconstruction loss.

The ceiling is measured: optimising rigid motion plus every torsion against a
steric objective takes atoms deeper than 0.5 A inside the receptor surface from
29.4% to 10.0%, against FLOWR's 7.3%. Rigid alone stops at 16.5% clash and
terminal-atom torsions alone fix 1% of buried atoms, so the interior torsions
are the degrees of freedom that matter.

Works off the bond list alone -- no RDKit -- so the training loop can build
these from what the pose-refine dataset already carries.
"""

from __future__ import annotations

from collections import deque

import numpy as np

#: Both endpoints must have this many heavy-atom neighbours for the bond to
#: move anything: a terminal bond rotates its single leaf atom on a cone, which
#: is measured not to help (1% of buried atoms fixed).
MIN_DEGREE = 2


def _adjacency(bonds: np.ndarray, n_atoms: int) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n_atoms)]
    for u, v in bonds[:, :2].astype(int):
        if 0 <= u < n_atoms and 0 <= v < n_atoms and u != v:
            adj[u].append(int(v))
            adj[v].append(int(u))
    return adj


def _reachable(
    adj: list[list[int]], start: int, blocked_edge: tuple[int, int]
) -> set[int]:
    """Nodes reachable from ``start`` without traversing ``blocked_edge``."""
    a, b = blocked_edge
    seen = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if (u == a and v == b) or (u == b and v == a):
                continue
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def _candidates(
    bonds: np.ndarray, adj: list[list[int]], n_atoms: int
) -> tuple[list[tuple[int, int]], list[np.ndarray]]:
    """Bonds whose removal disconnects the graph, with the moving side."""
    pairs: list[tuple[int, int]] = []
    masks: list[np.ndarray] = []
    seen: set[tuple[int, int]] = set()
    for u, v in bonds[:, :2].astype(int):
        i, j = (int(u), int(v)) if u < v else (int(v), int(u))
        if i == j or not (0 <= i < n_atoms and 0 <= j < n_atoms) or (i, j) in seen:
            continue
        seen.add((i, j))
        if len(adj[i]) < MIN_DEGREE or len(adj[j]) < MIN_DEGREE:
            continue
        side = _reachable(adj, j, (i, j))
        if i in side or not 0 < len(side) < n_atoms:
            continue  # in a ring, or nothing to move
        m = np.zeros(n_atoms, dtype=bool)
        m[sorted(side)] = True
        m[j] = False  # j lies on the axis
        if m.any():
            pairs.append((i, j))
            masks.append(m)
    return pairs, masks


def rotatable_bonds(
    bonds: np.ndarray,
    n_atoms: int,
    *,
    max_bonds: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotatable bonds and the atom mask each one rotates.

    A bond ``(i, j)`` is rotatable when cutting it splits the molecule (so it is
    not in a ring) and both endpoints have at least :data:`MIN_DEGREE`
    neighbours (so something other than a single leaf moves).

    Returns ``(pairs, masks)`` with shapes ``(K, 2)`` and ``(K, n_atoms)``.
    ``masks[k]`` is True for the atoms on ``pairs[k][1]``'s side, which are the
    ones a rotation about the axis moves. ``i`` and ``j`` themselves lie on the
    axis and never move.
    """
    if n_atoms <= 0 or bonds.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, n_atoms), dtype=bool)
    adj = _adjacency(np.asarray(bonds), n_atoms)
    pairs, masks = _candidates(np.asarray(bonds), adj, n_atoms)
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, n_atoms), dtype=bool)
    # Rotate the smaller side: identical geometry up to a rigid motion, and it
    # keeps the moved subtree small so a wrong angle disturbs fewer atoms.
    out_p: list[tuple[int, int]] = []
    out_m: list[np.ndarray] = []
    for (i, j), m in zip(pairs, masks, strict=True):
        if m.sum() * 2 > n_atoms:
            other = np.zeros(n_atoms, dtype=bool)
            other[:] = ~m
            other[i] = False
            other[j] = False
            if other.any():
                out_p.append((j, i))
                out_m.append(other)
                continue
        out_p.append((i, j))
        out_m.append(m)
    order = np.argsort([m.sum() for m in out_m])
    if max_bonds is not None:
        order = order[:max_bonds]
    return (
        np.array([out_p[k] for k in order], dtype=np.int64),
        np.stack([out_m[k] for k in order]).astype(bool),
    )


def _dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Signed dihedral p0-p1-p2-p3, in radians."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    n1 = np.cross(b0, b1)
    n2 = np.cross(b2, b1)
    m = np.cross(n1, b1 / max(np.linalg.norm(b1), 1e-8))
    x = float(n1 @ n2)
    y = float(m @ n2)
    return float(np.arctan2(y, x))


def torsion_delta(
    x0: np.ndarray,
    x1: np.ndarray,
    bonds: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    """Torsion rotation that takes ``x0``'s dihedrals to ``x1``'s, per pair.

    This is the target a torsion refiner should actually be given. Supervising
    it against a synthetic perturbation instead is not the same problem and is
    in fact unsolvable: the stored ``x0`` is already 0.76 A from the crystal, so
    "undo the twist I added" points back to that imperfect pose rather than to
    ``x1``, and nothing in the input says where the pre-twist pose was.

    Returns ``(K,)`` radians, 0 where a bond has no 4-atom dihedral.
    """
    n = x0.shape[0]
    adj = _adjacency(np.asarray(bonds), n)
    out = np.zeros(len(pairs), dtype=np.float32)
    for k, (i, j) in enumerate(np.asarray(pairs, dtype=int)):
        a = next((v for v in adj[i] if v != j), None)
        b = next((v for v in adj[j] if v != i), None)
        if a is None or b is None:
            continue
        d0 = _dihedral(x0[a], x0[i], x0[j], x0[b])
        d1 = _dihedral(x1[a], x1[i], x1[j], x1[b])
        # d0 - d1, not d1 - d0: applying angle t with
        # :func:`~prolit.model.torsion_transform.apply_torsions` DECREASES the
        # dihedral by t (its Rodrigues rotation runs the other way round from
        # the standard dihedral sign). Getting this backwards trains the head to
        # rotate away from the target, and nothing else in the pipeline notices.
        d = d0 - d1
        out[k] = np.arctan2(np.sin(d), np.cos(d))
    return out

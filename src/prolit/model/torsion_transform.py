"""Apply a (translation, rotation, torsions) transform to a ligand pose.

Differentiable in every parameter, so a network can be trained through it.

Why this output space rather than a free per-atom displacement: a rotation
about a bond axis preserves every interatomic distance except those spanning
the bond, and a rigid motion preserves all of them. So bond lengths and bond
angles are conserved **by construction**, not by a loss term. That matters
because the loss-term route is measured not to work here -- bonded pairs carry
0.077 of the reconstruction loss, and every free-displacement refiner in this
repository shortens bonds by ~0.12 A (10.0% of bonds out of tolerance without
one, 48.1% with ``refit_press0.6``), which costs PoseBusters validity
0.73 -> 0.42.

The reachable set is not a restriction that costs much: optimising rigid motion
plus every torsion against a steric objective takes atoms deeper than 0.5 A
inside the receptor from 29.4% to 10.0%, against FLOWR's measured 7.3%.
"""

from __future__ import annotations

import torch
from torch import Tensor


def rodrigues(v: Tensor, axis: Tensor, angle: Tensor, origin: Tensor) -> Tensor:
    """Rotate ``v`` (..., 3) about ``axis`` through ``origin`` by ``angle``."""
    a = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    p = v - origin
    c = torch.cos(angle)
    s = torch.sin(angle)
    dot = (p * a).sum(-1, keepdim=True)
    turned = p * c + torch.cross(a.expand_as(p), p, dim=-1) * s + a * dot * (1 - c)
    return origin + turned


def apply_torsions(
    pos: Tensor,  # (N, 3)
    pairs: Tensor,  # (K, 2) long: (i, j); the axis runs i -> j
    masks: Tensor,  # (K, N) bool: atoms moved by torsion k
    angles: Tensor,  # (K,)
) -> Tensor:
    """Rotate each subtree about its bond axis, in order.

    Applied sequentially because the subtrees nest: rotating an outer bond after
    an inner one must act on the already-rotated coordinates, which is what a
    real torsional degree of freedom does.
    """
    out = pos
    for k in range(pairs.shape[0]):
        i, j = int(pairs[k, 0]), int(pairs[k, 1])
        axis = out[j] - out[i]
        moved = rodrigues(out, axis, angles[k], out[j])
        out = torch.where(masks[k][:, None], moved, out)
    return out


def apply_transform(  # noqa: PLR0913
    pos: Tensor,  # (N, 3) ligand coordinates
    translation: Tensor,  # (3,)
    rot_vec: Tensor,  # (3,) axis-angle; magnitude is the angle
    pairs: Tensor,  # (K, 2)
    masks: Tensor,  # (K, N)
    angles: Tensor,  # (K,)
) -> Tensor:
    """Torsions first, then a rigid rotation about the centroid, then translate.

    Order matters only for interpretation, not expressiveness: torsions are
    defined on the molecule's own frame, so applying them before the global
    motion keeps the predicted angles independent of where the pose currently
    sits.
    """
    x = apply_torsions(pos, pairs, masks, angles) if pairs.numel() else pos
    centroid = x.mean(dim=0, keepdim=True)
    theta = rot_vec.norm().clamp_min(1e-8)
    x = rodrigues(x, rot_vec, theta, centroid)
    return x + translation

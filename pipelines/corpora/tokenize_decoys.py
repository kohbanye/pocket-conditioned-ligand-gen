"""Build a decoy-pose training set (complex tokens + RMSD) for the scoring head.

Zero-shot PLL already ranks native above decoy (35% -> 55% docking power as the
MLM trains); a discriminative head trained on RMSD-labelled decoys should push
further (the Interformer / DeepBSP recipe). We generate decoys WITHOUT docking
software or re-downloading CrossDocked types: take CASF-excluded BioLIP native
complexes and rigidly perturb the ligand (rotation about its centroid +
translation) by a graded magnitude, so the RMSD to the crystal pose is known
exactly. The protein pocket is fixed, so only the ligand codes change -- exactly
the P(ligand pose | pocket) signal the head must learn to score.

Output: ``{split}.bin`` (uint16 tokens) + ``{split}.len`` + ``{split}.rmsd``
(float32, one per doc). Native pose is RMSD 0.

Run (single GPU)::

    uv run python pipelines/corpora/tokenize_decoys.py \
        --ckpt "<atom-vqvae>.ckpt" \
        --norm-stats data/descriptor_cache_allatom/normalization_stats.pt \
        --n-complexes 12000 --n-decoys 16 --out-dir data/lm_tokens_decoys
"""

from __future__ import annotations

import argparse
import functools
import gzip
import json
import logging
import signal
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

# Sibling modules in this directory, imported by bare name: Python puts a
# script's own directory on sys.path[0], so this resolves from any cwd.
from tokenize_biolip import (
    _bucket_code,
    _cd_test_pdbs,
    _load_ccd_smiles,
    _parse_biolip_txt,
    _read_needed,
)

from prolit.config import AtomVQVAETrainingConfig, PocketExtractionConfig
from prolit.seeding import add_seed_argument, seed_from_args
from prolit.tokenizers.geometry import random_rotation_matrix
from prolit.tokenizers.ligand import parse_ligand_pdb_text
from prolit.tokenizers.lm_vocab import AtomLMVocab
from prolit.tokenizers.loaders import load_atom_vqvae
from prolit.tokenizers.pose_encoder import PoseEncoder

logging.basicConfig(level=logging.INFO)
if TYPE_CHECKING:
    from collections.abc import Callable

    from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix for a unit ``axis`` and ``angle`` (radians)."""
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def _perturb(
    coords: np.ndarray, rng: np.random.Generator, scale: float
) -> tuple[np.ndarray, float]:
    """Rigidly rotate+translate heavy-atom coords; return (new_coords, RMSD)."""
    centroid = coords.mean(0)
    angle = rng.uniform(0, scale * np.pi / 2)  # up to 90 deg at scale=1
    trans = rng.normal(size=3)
    trans = trans / (np.linalg.norm(trans) + 1e-9) * rng.uniform(0, scale * 6.0)
    rot = _rotation(rng.normal(size=3), angle)
    new = (coords - centroid) @ rot.T + centroid + trans
    rmsd = float(np.sqrt(((new - coords) ** 2).sum(axis=1).mean()))
    return new, rmsd


#: Cap on the automorphism search; a few ligands have thousands and the
#: symmetry-corrected RMSD is already converged long before that.
_MAX_AUTOMORPHISMS = 2000

#: How much rigid motion rides on top of the torsions in
#: :func:`_near_torsion_perturb`, as a fraction of the torsion RMSD. Tuned to
#: reproduce CASF's 0-1 A displacement CV of 0.64; see that function's table.
_NEAR_RIGID_ALPHA = 0.3


def _automorphisms(atoms: list, bonds: list, hidx: np.ndarray) -> np.ndarray | None:
    """Heavy-atom permutations that leave the ligand graph unchanged.

    Bond orders are dropped so resonance-equivalent atoms (a phosphate's three
    oxygens, a carboxylate's two) come out equivalent, which is what a
    symmetry-corrected RMSD needs; keeping them splits those atoms apart and
    the correction silently does nothing on exactly the ligands that need it.
    """
    from rdkit import Chem  # noqa: PLC0415

    m = Chem.RWMol()
    for a in atoms:
        m.AddAtom(Chem.Atom(a[0].split(".")[0]))
    seen: set[tuple[int, int]] = set()
    for b in bonds:
        i, j = int(b[0]), int(b[1])
        if (i, j) in seen or (j, i) in seen:
            continue
        seen.add((i, j))
        m.AddBond(i, j, Chem.BondType.SINGLE)
    mol = m.GetMol()
    try:
        matches = mol.GetSubstructMatches(
            mol, uniquify=False, useChirality=False, maxMatches=_MAX_AUTOMORPHISMS
        )
    except Exception:  # noqa: BLE001  (RDKit raises bare exceptions here)
        return None
    pos = {a: k for k, a in enumerate(hidx.tolist())}
    out = [
        [pos[mm[a]] for a in hidx.tolist()]
        for mm in matches
        if all(mm[a] in pos for a in hidx.tolist())
    ]
    return np.asarray(out, dtype=np.int64) if out else None


def _pocket_tree(
    pdb_text: str, centre: np.ndarray, radius: float = 16.0
) -> cKDTree | None:
    """KD-tree over the heavy protein atoms near the ligand, or None."""
    xs = []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        element = line[76:78].strip() or line[12:16].strip()[:1]
        if element == "H":
            continue
        try:
            xs.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except ValueError:
            continue
    if not xs:
        return None
    from scipy.spatial import cKDTree  # noqa: PLC0415

    arr = np.asarray(xs)
    near = arr[np.linalg.norm(arr - centre, axis=1) < radius]
    return cKDTree(near) if len(near) else None


def _spin(
    coords: np.ndarray,
    hidx: np.ndarray,
    rng: np.random.Generator,
    lo_deg: float,
    hi_deg: float,
) -> np.ndarray:
    """Turn the ligand in place: same site, wrong way round.

    :func:`_perturb` always rides a translation of up to 6 A on its rotation, so
    the corpus holds no pose that sits in the right place facing the wrong way
    -- and that is the pose docking produces most. Measured on the trained head:
    predicted RMSD climbs with rotation only to ~60-90 deg and then FALLS, so a
    fully flipped ligand (true RMSD 5.80 A) scores 2.86 A, better than a 45 deg
    one, while pure translation tracks the truth to 5.89 vs 5.80. The failure is
    specific to orientation, and it is a hole in the training distribution
    rather than a limit of the tokens.

    The axis is the ligand's own longest principal axis (jittered): turning an
    elongated ligand about its long axis roughly preserves the volume it
    occupies, which clashes 0.48 times per heavy atom against 0.99 for a random
    axis. :func:`_place` then slides it to fit.
    """
    heavy = coords[hidx]
    centred = heavy - heavy.mean(0)
    _, vecs = np.linalg.eigh(centred.T @ centred)
    axis = vecs[:, -1] + rng.normal(size=3) * 0.15
    rot = _rotation(axis, float(np.radians(rng.uniform(lo_deg, hi_deg))))
    centroid = coords.mean(0)
    return (coords - centroid) @ rot.T + centroid


def _clash_per_atom(ligand: np.ndarray, pocket: cKDTree | None) -> float:
    """Protein heavy atoms within 2.6 A of a ligand atom, per ligand atom.

    ``pocket`` is a :class:`scipy.spatial.cKDTree`; the placement loop calls
    this a few hundred times per decoy and a dense distance matrix there costs
    more than the tokenizer.
    """
    if pocket is None:
        return 0.0
    return sum(len(h) for h in pocket.query_ball_point(ligand, 2.6)) / len(ligand)


#: Clashes per ligand heavy atom a decoy may keep. CASF's own docking decoys
#: sit at 0.03-0.10 across every RMSD band and the crystal pose at 0.000.
_MAX_CLASH = 0.15

#: A protein heavy atom this close to a ligand heavy atom counts as a clash.
_CLASH_RADIUS = 2.6


def _place(
    coords: np.ndarray,
    hidx: np.ndarray,
    pocket: cKDTree | None,
    rng: np.random.Generator,
    n_offsets: int = 12,
) -> tuple[np.ndarray, float]:
    """Slide a pose to the least-clashing offset near where it was put.

    A docking program only ever proposes poses that fit, so every decoy CASF
    scores is clash-free -- 0.083 clashes per heavy atom at 3-6 A RMSD. The
    perturbation classes here are not: ``_perturb`` sits at 1.622 and
    ``_conf_perturb`` at 1.513 in that band, ~20x worse, because a random
    displacement drives the ligand straight into the protein. A head trained on
    that learns "wrong pose = clashing pose", which separates nothing at test
    time, and it is why every loss change so far moved DP@1A (whose band, under
    1.5 A, IS realistic at 0.098 vs 0.033) and left DP@2A flat.

    This does what docking does with a placement it wants to keep: nudge it to
    the nearby offset that fits best. Returns the pose and its clash count so
    the caller can resample when nothing fits.
    """
    if pocket is None:
        return coords, 0.0
    best, best_clash = coords, _clash_per_atom(coords[hidx], pocket)
    if best_clash <= _MAX_CLASH:
        return best, best_clash
    for off in rng.normal(size=(n_offsets, 3)) * 0.9:
        cand = coords + off
        clash = _clash_per_atom(cand[hidx], pocket)
        if clash < best_clash:
            best, best_clash = cand, clash
        if clash <= _MAX_CLASH:
            break
    return best, best_clash


def _placed(
    make: Callable[[], np.ndarray | None],
    hidx: np.ndarray,
    pocket: cKDTree | None,
    rng: np.random.Generator,
    tries: int = 5,
) -> np.ndarray | None:
    """Draw poses from ``make`` until one fits the pocket; keep the best try."""
    best: np.ndarray | None = None
    best_clash = float("inf")
    for _ in range(tries):
        cand = make()
        if cand is None:
            continue
        cand, clash = _place(cand, hidx, pocket, rng)
        if clash < best_clash:
            best, best_clash = cand, clash
        if clash <= _MAX_CLASH:
            break
    return best



def _mol_with_conformer(atoms: list, bonds: list):  # noqa: ANN202
    """RDKit mol carrying ``atoms`` as its conformer, or ``None`` if it fails.

    Shared by the two torsion perturbers so they agree on how a parsed ligand
    becomes something ``rdMolTransforms`` can rotate.
    """
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Geometry import Point3D  # noqa: PLC0415

    bt = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    rw = Chem.RWMol()
    for a in atoms:
        try:
            rw.AddAtom(Chem.Atom(a[0]))
        except Exception:  # noqa: BLE001
            rw.AddAtom(Chem.Atom("C"))
    n = len(atoms)
    for i, j, t in bonds:
        if 0 <= i < n and 0 <= j < n and i != j:
            try:
                rw.AddBond(i, j, bt.get(t, Chem.BondType.SINGLE))
            except Exception:  # noqa: BLE001, S112
                continue
    mol = rw.GetMol()
    conf = Chem.Conformer(n)
    for i, a in enumerate(atoms):
        conf.SetAtomPosition(i, Point3D(float(a[1]), float(a[2]), float(a[3])))
    mol.AddConformer(conf)
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
    except Exception:  # noqa: BLE001
        return None
    return mol


def _rotatable(mol) -> tuple:  # noqa: ANN001
    """Rotatable single bonds as ``(i, j)`` index pairs."""
    from rdkit import Chem  # noqa: PLC0415

    patt = Chem.MolFromSmarts("[!$(*#*)&!D1]-!@[!$(*#*)&!D1]")
    return mol.GetSubstructMatches(patt) if patt is not None else ()


def _near_torsion_perturb(  # noqa: C901, PLR0913
    atoms: list,
    bonds: list,
    hidx: np.ndarray,
    base: np.ndarray,
    rng: np.random.Generator,
    target: float,
) -> np.ndarray | None:
    """Torsion-only decoy landing near ``target`` A RMSD, nothing else moved.

    This is the class the 2026-08-20 failure analysis says is missing. Measured
    on CASF, the coefficient of variation of per-atom displacement in the
    decisive 0-1 A band is 0.64 -- most atoms essentially perfect, a group swung
    out -- while this corpus's 0-1 A poses sit at 0.40, i.e. the whole molecule
    nudged. The head therefore never had to tell "right place, wrong torsion"
    from "right place, right torsion", which is what 90% of its CASF mistakes
    are (picked and correct poses differ by 2.41 A of internal geometry against
    0.88 A of centroid).

    **Several torsions plus a little rigid, all measured against CASF.** The
    three knobs were tuned by generating poses off CASF ligands and matching
    the 0-1 A CV:

    | recipe | CV in 0-1 A |
    |---|---|
    | one torsion only | 1.53 |
    | subset of torsions, no rigid | 0.97 |
    | **subset + rigid at alpha=0.3** | **0.66** |
    | subset + rigid at alpha=0.5 | 0.53 |
    | rigid only (``_perturb``) | 0.33 |
    | CASF, measured | **0.64** |

    Turning a single bond leaves every atom on its near side exactly put and
    swings the far side alone -- as lopsided as the rigid class is uniform, and
    just as unlike CASF. ``_conf_perturb`` mixes the same two ingredients but at
    a fixed scale, so it lands in 0-1 A only 23% of the time; bisecting to a
    requested RMSD is what puts the mass where the decision is made.
    """
    from rdkit.Chem import rdMolTransforms as rmt  # noqa: PLC0415, N813

    mol = _mol_with_conformer(atoms, bonds)
    if mol is None:
        return None
    rot = _rotatable(mol)
    if not rot:
        return None
    conf = mol.GetConformer()
    n = len(atoms)

    picked = []
    for b1, b2 in rot:
        if rng.random() > 0.5 and len(rot) > 1:  # noqa: PLR2004
            continue
        nb1 = mol.GetAtomWithIdx(b1).GetNeighbors()
        nb2 = mol.GetAtomWithIdx(b2).GetNeighbors()
        a1 = [x.GetIdx() for x in nb1 if x.GetIdx() != b2]
        a4 = [x.GetIdx() for x in nb2 if x.GetIdx() != b1]
        if not a1 or not a4:
            continue
        try:
            cur = rmt.GetDihedralDeg(conf, a1[0], b1, b2, a4[0])
        except Exception:  # noqa: BLE001, S112
            continue
        picked.append((a1[0], b1, b2, a4[0], cur, float(rng.uniform(-1.0, 1.0))))
    if not picked:
        return None

    def at(scale: float) -> np.ndarray | None:
        for i, j, k, m, cur, w in picked:
            try:
                rmt.SetDihedralDeg(conf, i, j, k, m, cur + scale * w)
            except Exception:  # noqa: BLE001, S112
                continue
        return np.array([list(conf.GetAtomPosition(t)) for t in range(n)])

    # Bisect torsions to the share of the target they should carry, so the
    # rigid part added below brings the total back to ``target``.
    tors_target = target / np.sqrt(1.0 + _NEAR_RIGID_ALPHA**2)
    lo, hi, best = 0.0, 180.0, None
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        new = at(mid)
        if new is None:
            return None
        best = new
        r = float(np.sqrt(((new[hidx] - base[hidx]) ** 2).sum(axis=1).mean()))
        if r < tors_target:
            lo = mid
        else:
            hi = mid
    if best is None:
        return None
    rt = float(np.sqrt(((best[hidx] - base[hidx]) ** 2).sum(axis=1).mean()))
    shift = rng.normal(size=3)
    shift = shift / (np.linalg.norm(shift) + 1e-9) * _NEAR_RIGID_ALPHA * rt
    spin = _rotation(rng.normal(size=3), rng.normal() * _NEAR_RIGID_ALPHA * 0.15)
    centre = best.mean(0)
    return (best - centre) @ spin.T + centre + shift


def _mixed_perturb(
    atoms: list,
    bonds: list,
    rng: np.random.Generator,
    r_scale: float,
    t_scale: float,
) -> np.ndarray | None:
    """Torsions at ``t_scale`` and a rigid move at ``r_scale``, INDEPENDENTLY.

    The existing generators tie the two together: ``_perturb`` is rigid with no
    torsion at all, ``_conf_perturb`` turns torsions and adds a rigid move fixed
    at 0.4x its own scale. A decoy is therefore either displaced or distorted,
    never a free mixture, and at any given RMSD the two components are mutually
    exclusive. The head exploits that -- measured on CASF with RMSD held fixed
    inside each complex, its score correlates -0.474 with centroid displacement
    and +0.484 with deviation of the internal conformer, while RTMScore sits at
    +0.03 on both. It has learned rigid displacement as the signal for badness
    and barely penalises torsion, so among real docked poses, where the two
    always co-occur, it prefers the ones that are distorted rather than moved.

    Sampling the magnitudes independently puts every mixture at every RMSD, so
    neither component can stand in for the other.
    """
    from rdkit.Chem import rdMolTransforms as rmt  # noqa: PLC0415, N813

    mol = _mol_with_conformer(atoms, bonds)
    if mol is None:
        return None
    conf = mol.GetConformer()
    n = len(atoms)
    if t_scale > 0:
        for b1, b2 in _rotatable(mol):
            if rng.random() > 0.5:  # noqa: PLR2004
                continue
            nb1 = mol.GetAtomWithIdx(b1).GetNeighbors()
            nb2 = mol.GetAtomWithIdx(b2).GetNeighbors()
            a1 = [x.GetIdx() for x in nb1 if x.GetIdx() != b2]
            a4 = [x.GetIdx() for x in nb2 if x.GetIdx() != b1]
            if not a1 or not a4:
                continue
            try:
                cur = rmt.GetDihedralDeg(conf, a1[0], b1, b2, a4[0])
                rmt.SetDihedralDeg(
                    conf, a1[0], b1, b2, a4[0],
                    cur + rng.uniform(-t_scale * 180, t_scale * 180),
                )
            except Exception:  # noqa: BLE001, S112
                continue
    new = np.array([list(conf.GetAtomPosition(i)) for i in range(n)])
    if r_scale > 0:
        new = _perturb(new, rng, r_scale)[0]
    return new


def _conf_perturb(
    atoms: list, bonds: list, rng: np.random.Generator, scale: float
) -> np.ndarray | None:
    """Rotate a random subset of rotatable-bond torsions + a small rigid shift.

    Produces a VALID-geometry conformational decoy (like a real docking pose that
    is near-native in place but wrong in torsions), which rigid perturbation
    alone cannot -- closing the train/test decoy-distribution gap. Returns the
    perturbed (all-atom) coords, or ``None`` if the RDKit build fails.
    """
    from rdkit.Chem import rdMolTransforms as rmt  # noqa: PLC0415, N813

    mol = _mol_with_conformer(atoms, bonds)
    if mol is None:
        return None
    n = len(atoms)
    rot = _rotatable(mol)
    c = mol.GetConformer()
    for b1, b2 in rot:
        if rng.random() > 0.5:  # noqa: PLR2004
            continue
        a1 = [
            x.GetIdx()
            for x in mol.GetAtomWithIdx(b1).GetNeighbors()
            if x.GetIdx() != b2
        ]
        a4 = [
            x.GetIdx()
            for x in mol.GetAtomWithIdx(b2).GetNeighbors()
            if x.GetIdx() != b1
        ]
        if not a1 or not a4:
            continue
        try:
            cur = rmt.GetDihedralDeg(c, a1[0], b1, b2, a4[0])
            rmt.SetDihedralDeg(
                c, a1[0], b1, b2, a4[0], cur + rng.uniform(-scale * 180, scale * 180)
            )
        except Exception:  # noqa: BLE001, S112
            continue
    new = np.array([list(c.GetAtomPosition(i)) for i in range(n)], dtype=np.float64)
    return _perturb(new, rng, scale * 0.4)[0]  # small rigid on top


def _kabsch(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid transform (rot, trans) that best superposes ``p`` onto ``q``."""
    pc, qc = p.mean(0), q.mean(0)
    u, _, vt = np.linalg.svd((p - pc).T @ (q - qc))
    d = float(np.sign(np.linalg.det(vt.T @ u.T)))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, qc - rot @ pc


def _conformer_decoys(  # noqa: PLR0913
    atoms: list, bonds: list, hidx: np.ndarray, base: np.ndarray, n: int, seed: int
) -> list[np.ndarray]:
    """Freshly embedded conformers, rigidly superposed onto the native pose.

    The decoys a docking program produces near the native site have *correct
    placement but an independently generated internal conformer*; perturbing the
    crystal conformer (rigidly or by torsion) never yields that. A head trained
    only on perturbed-crystal decoys can therefore learn "the exact crystal
    conformer is the native one" and bury a genuinely good redocked pose -- the
    observed failure mode (good poses ranked >10 on 14 CASF targets). These
    decoys put realistic 0.5-2.5 A near-natives in the training distribution.
    """
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    bt = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    rw = Chem.RWMol()
    for a in atoms:
        try:
            rw.AddAtom(Chem.Atom(a[0]))
        except Exception:  # noqa: BLE001
            rw.AddAtom(Chem.Atom("C"))
    na = len(atoms)
    for i, j, t in bonds:
        if 0 <= i < na and 0 <= j < na and i != j:
            try:
                rw.AddBond(i, j, bt.get(t, Chem.BondType.SINGLE))
            except Exception:  # noqa: BLE001, S112
                continue
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001
        return []  # caller falls back to the perturbation decoys
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.3
    params.maxIterations = 200
    try:
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=3 * n, params=params)
    except Exception:  # noqa: BLE001
        return []
    poses = []
    for cid in cids:
        conf = mol.GetConformer(cid)
        new = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(na)], dtype=np.float64
        )
        rot, trans = _kabsch(new[hidx], base[hidx])
        new = new @ rot.T + trans
        poses.append(
            (float(np.sqrt(((new[hidx] - base[hidx]) ** 2).sum(1).mean())), new)
        )
    if not poses:
        return []
    # A docking run *searches* conformers, so its pose set is graded: the best
    # sampled conformer sits near the native and the rest fan out. A random
    # embedding is not -- it lands 2-4 A away. Sample the pool evenly along its
    # own RMSD order so the closest conformer is always kept and the class spans
    # near-native to clearly-wrong, like a real pose set.
    poses.sort(key=lambda t: t[0])
    idx = np.unique(np.linspace(0, len(poses) - 1, num=n).round().astype(int))
    return [poses[i][1] for i in idx]


def _decompose(new: np.ndarray, base: np.ndarray) -> tuple[float, float]:
    """Split a pose's deviation into translation and internal-conformer parts.

    RMSD^2 is the sum of three orthogonal terms: the centroid offset, the
    residual rotation of the centred coordinates, and what optimal
    superposition cannot remove -- the internal conformer difference. The first
    two are how the ligand sits in the pocket; the third is what shape it took.
    Returns (translation, internal); the rotation term is the remainder, so a
    consumer recovers it as ``sqrt(rmsd^2 - t^2 - i^2)`` and the three always
    reconstruct the RMSD exactly.
    """
    t = new.mean(axis=0) - base.mean(axis=0)
    a = new - new.mean(axis=0)
    b = base - base.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = float(np.sign(np.linalg.det(u @ vt)))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    internal = float(np.sqrt(((a @ rot - b) ** 2).sum(axis=1).mean()))
    return float(np.linalg.norm(t)), internal


def _record_decoy(  # noqa: PLR0913
    new: np.ndarray,
    mol: dict,
    base: np.ndarray,
    hidx: np.ndarray,
    out: tuple[list, list, list],
    perms: np.ndarray | None = None,
) -> None:
    """Append one decoy's coordinates and its three label streams.

    ``perms`` are the ligand's heavy-atom automorphisms; when given, the atom
    mapping that makes this pose look most like the native is used, which is
    the RMSD CASF reports. Only the spin class needs it -- the other classes
    are continuous deformations of the native, where the identity mapping is
    already optimal (measured: median naive-minus-symmetric 0.000 A).
    """
    mols, rmsds, comps_and_disps = out
    comps, disps = comps_and_disps
    nb, bb = new[hidx], base[hidx]
    order = None
    if perms is not None:
        k = int(np.argmin(((nb[perms] - bb) ** 2).sum(2).mean(1)))
        order = perms[k]
        nb = nb[order]
    d = np.linalg.norm(nb - bb, axis=1)
    if order is not None:
        # d[i] is how far POSE atom order[i] sits from native atom i, and the
        # tokens run in pose order, so scatter it back before storing.
        scattered = np.empty_like(d)
        scattered[order] = d
        d = scattered
    disps.append(d.astype(np.float32))
    rmsds.append(float(np.sqrt((d**2).mean())))
    comps.append(_decompose(nb, bb))
    mols.append({
        "atoms": [
            (a[0], float(new[i][0]), float(new[i][1]), float(new[i][2]))
            for i, a in enumerate(mol["atoms"])
        ],
        "bonds": mol["bonds"],
    })


def _ccd_heavy_atoms(path: Path) -> dict[str, int]:
    """Heavy-atom count per CCD, read off the formula column of ligand.tsv.gz.

    The SMILES column holds several tautomers separated by ``;`` and does not
    parse as one molecule; the formula does, and it is what the per-CCD cap
    needs to know which ligands are large enough to be worth capping.
    """
    import gzip  # noqa: PLC0415
    import re  # noqa: PLC0415

    out: dict[str, int] = {}
    with gzip.open(path, "rt") as fh:
        next(fh, None)
        for line in fh:
            parts = line.split("\t")
            if len(parts) < 2:  # noqa: PLR2004
                continue
            n = 0
            for element, count in re.findall(r"([A-Z][a-z]?)\s*(\d*)", parts[1]):
                if element in ("", "H"):
                    continue
                n += int(count) if count else 1
            if n:
                out[parts[0]] = n
    return out


def _decoy_drawer(  # noqa: PLR0913
    k: int,
    n_near: int,
    n_pert: int,
    confs: list,
    mol: dict,
    base: np.ndarray,
    hidx: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> Callable[[], np.ndarray | None]:
    """A zero-argument sampler for decoy slot ``k``, redrawable by :func:`_placed`.

    A factory rather than closures written inline in the loop: the loop rebinds
    ``mol``/``base``/``hidx`` every complex, and a closure over those would read
    whatever they happen to be when it is finally called.
    """

    def near_torsion() -> np.ndarray | None:
        tgt = float(rng.uniform(args.near_torsion_lo, args.near_torsion_hi))
        out = _near_torsion_perturb(mol["atoms"], mol["bonds"], hidx, base, rng, tgt)
        return _perturb(base, rng, 0.1)[0] if out is None else out

    def conformer() -> np.ndarray | None:
        return confs[k - n_near - n_pert]

    def perturbed() -> np.ndarray | None:
        scale = (k - n_near + 1) / max(1, n_pert)
        if args.mixed_perturb:
            rs = float(rng.uniform(0.0, 1.0)) * scale
            ts = float(rng.uniform(0.0, 1.0)) * scale
            out = _mixed_perturb(mol["atoms"], mol["bonds"], rng, rs, ts)
        elif k % 2 == 0:
            out = _perturb(base, rng, scale)[0]
        else:
            out = _conf_perturb(mol["atoms"], mol["bonds"], rng, scale)
        return _perturb(base, rng, scale)[0] if out is None else out

    if k < n_near:
        return near_torsion
    if k >= n_near + n_pert:
        return conformer
    return perturbed


def _load_stapled_vocab(path: Path | None, confseq_vocab_cls: type):  # noqa: ANN202
    """Read the frozen ConfSeq vocabulary that a stapled corpus was built with."""
    from prolit.tokenizers.stapled import StapledVocab  # noqa: PLC0415

    if path is None or not path.exists():
        msg = (
            "--stapled-esm3-cache needs --stapled-vocab pointing at a vocabulary "
            "built by --build-stapled-vocab; the ids are baked into every stream "
            "and rebuilding them per shard would give one corpus two alphabets"
        )
        raise SystemExit(msg)
    return StapledVocab(confseq=confseq_vocab_cls.load(path))


def _emit_stapled(  # noqa: PLR0913
    stapled: Any,  # noqa: ANN401
    pocket: Any,  # noqa: ANN401
    mols: list[dict],
    rmsds: list[float],
    comps: list[tuple[float, float]],
    hidx: np.ndarray,
    writer: Any,  # noqa: ANN401
    symbol_counts: dict[str, int],
    tally: dict[str, int],
    *,
    build_vocab_only: bool,
) -> int:
    """Write one site's poses as stapled streams. Returns 1 if any landed.

    No rotation augmentation, unlike the ProLIT arm. ESM3's codes and ConfSeq's
    tokens are both invariant, so re-expressing the complex in a rotated frame
    would move nothing but the placement grid -- N near-identical copies of one
    document rather than N views of it. That asymmetry is a property of the
    representations and belongs in the write-up, not hidden by manufacturing
    duplicates here.

    Per-atom displacement labels are not written either: the stapled ligand
    block is ConfSeq tokens, which do not correspond one-to-one with heavy
    atoms, so the per-atom head has nothing to attach to. The RMSD label and the
    translation/rotation decomposition are unaffected.
    """
    wrote = 0
    for m_, rmsd, cmp_ in zip(mols, rmsds, comps, strict=True):
        tally["poses"] = tally.get("poses", 0) + 1
        if build_vocab_only:
            toks = stapled.confseq_tokens(m_["atoms"], m_["bonds"], hidx)
            if toks is None:
                tally["confseq_failed"] = tally.get("confseq_failed", 0) + 1
                continue
            for t in toks:
                symbol_counts[t] = symbol_counts.get(t, 0) + 1
            wrote += 1
            continue
        seq, why = stapled.ligand_seq_with_reason(
            pocket, m_["atoms"], m_["bonds"], hidx
        )
        if seq is None:
            # Counted by reason, not lumped together. "confseq" is a real limit
            # of the baseline and belongs in the paper; "oov" is a build error
            # -- the frozen alphabet was collected from too small a sample --
            # and means this corpus is silently missing good data.
            tally[f"failed_{why}"] = tally.get(f"failed_{why}", 0) + 1
            continue
        writer.write(seq, rmsd, None, cmp_)
        wrote += 1
    return 1 if wrote else 0


class _SiteBudget:
    """Abandon one ligand rather than the shard it is in.

    A single pathological molecule can wedge a whole shard: RDKit's
    ``GetSubstructMatches`` over a highly symmetric ligand explores a factorial
    space, and ``maxMatches`` caps the results it keeps, not the search it
    does. One shard spun a core for eleven hours that way and produced nothing
    after its first six.

    SIGALRM, not a process pool, because this loop is single-threaded and a
    pool would restructure the file. The limit of that choice is worth being
    explicit about: a Python signal handler runs between bytecodes, so a call
    that stays inside C for hours is **not** interrupted by this. It catches
    the interruptible majority; the meta checkpoint after every bucket is what
    covers the rest, by making a wedged shard's finished work readable anyway.

    The raised ``TimeoutError`` is an ``OSError``, so the per-site
    ``except Exception`` already skips the complex -- this only has to make
    the clock run.
    """

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self.hit = 0
        if seconds > 0:
            signal.signal(signal.SIGALRM, self._raise)

    def arm(self) -> None:
        if self.seconds > 0:
            signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def disarm(self) -> None:
        if self.seconds > 0:
            signal.setitimer(signal.ITIMER_REAL, 0)

    def _raise(self, _signum: int, _frame: object) -> None:
        self.hit += 1
        msg = f"complex exceeded its {self.seconds}s budget"
        raise TimeoutError(msg)


def _write_meta(  # noqa: PLR0913
    out_dir: Path,
    args: argparse.Namespace,
    vocab: Any,  # noqa: ANN401
    stapled: Any,  # noqa: ANN401
    n_ok: int,
    tally: dict[str, int],
    writers: dict[str, Any],
) -> dict:
    """Describe what is on disk right now, and put it there.

    Called after every bucket, not only at the end. A shard that wedges on one
    pathological ligand -- or that a walltime kills -- has all of its finished
    documents written, but without a meta.json ``concat_decoy_shards.py``
    cannot read them and 11 hours of work is lost. That happened: one shard
    stopped producing at 04:55, spun a core until its 12-hour cap, and its
    21,760 documents went in the bin. Rewriting a 700-byte JSON per bucket is
    the cheapest insurance in this file.
    """
    for w in writers.values():
        w.flush()
    meta: dict = {
        "vocab_size": (
            stapled.vocab.vocab_size if stapled is not None else vocab.vocab_size
        ),
        # Separate-tokenizers mode doubles the code space (protein then ligand).
        "atom_codebook_size": (
            2 * args.codebook_size
            if args.separate_protein_ckpt is not None
            else args.codebook_size
        ),
        "source": "biolip2_rigid_decoys",
        "n_decoys": args.n_decoys,
        "n_near_torsion_decoys": args.n_near_torsion_decoys,
        "complexes_used": n_ok,
        "splits": {},
    }
    if args.separate_protein_ckpt is not None:
        meta["separate_tokenizers"] = True
    if stapled is not None:
        # A reader has to be able to tell this corpus from ProLIT's without
        # opening it: the vocabularies are different sizes, the ligand block is
        # not one token per atom, and the placement rides in the first tokens
        # of that block. A trainer pointed at the wrong one would still run.
        meta["stapled"] = {
            "protein_tokenizer": "esm3_structure_v0",
            "ligand_tokenizer": "confseq",
            "pose_bits": stapled.vocab.pose_bits,
            "n_pose_tokens": stapled.vocab.n_pose_tokens,
            "confseq_vocab": stapled.vocab.confseq.size,
            "esm3_cache": str(args.stapled_esm3_cache),
            "confseq_vocab_path": str(args.stapled_vocab),
            "coverage": dict(tally),
        }
    for split, w in writers.items():
        meta["splits"][split] = {
            "num_docs": w.num_docs,
            "num_tokens": w.num_tokens,
            "max_len": w.max_len,
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _dump_bucket(
    out_dir: Path,
    code: str,
    site_list: list[tuple],
    receptors: dict[str, bytes],
) -> int:
    """Write one bucket's receptor structures as gzipped JSON lines.

    One file per bucket, not per receptor: a BioLiP pass touches tens of
    thousands of structures and the group filesystem has a limited inode
    budget. Deduplicated by receptor id, since many sites share a chain.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"receptors_{code}.jsonl.gz"
    seen: set[str] = set()
    n = 0
    with gzip.open(path, "wt") as fh:
        for pdb, rchain, *_ in site_list:
            sid = f"{pdb}{rchain}"
            if sid in seen:
                continue
            raw = receptors.get(f"{sid}.pdb")
            if raw is None:
                continue
            seen.add(sid)
            fh.write(
                json.dumps({"id": sid, "pdb": raw.decode("utf-8", "replace")}) + "\n"
            )
            n += 1
    return n


class _RmsdWriter:
    """Streams tokens (.bin/.len), one float32 RMSD per doc (.rmsd), and the
    per-ligand-atom displacement (.disp float32, ``.dlen`` uint16 counts).

    The per-atom stream is dense supervision: one RMSD scalar tells the head only
    that a pose is wrong, while a displacement per ligand token tells it *which
    atoms* are misplaced -- ~30 labels per pose instead of 1, and exactly the
    quantity the pose score aggregates.
    """

    def __init__(self, out_dir: Path, split: str) -> None:
        self._bin = (out_dir / f"{split}.bin").open("wb")
        self._len = (out_dir / f"{split}.len").open("wb")
        self._rmsd = (out_dir / f"{split}.rmsd").open("wb")
        self._disp = (out_dir / f"{split}.disp").open("wb")
        self._comp = (out_dir / f"{split}.comp").open("wb")
        self._dlen = (out_dir / f"{split}.dlen").open("wb")
        self.num_docs = 0
        self.num_tokens = 0
        self.max_len = 0

    def write(
        self,
        seq: list[int],
        rmsd: float,
        disp: np.ndarray | None = None,
        comp: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        arr = np.asarray(seq, dtype=np.uint16)
        self._bin.write(arr.tobytes())
        self._len.write(np.asarray([len(seq)], dtype=np.uint16).tobytes())
        self._rmsd.write(np.asarray([rmsd], dtype=np.float32).tobytes())
        d = np.asarray([] if disp is None else disp, dtype=np.float32)
        self._disp.write(d.tobytes())
        self._dlen.write(np.asarray([d.shape[0]], dtype=np.uint16).tobytes())
        self._comp.write(np.asarray(comp, dtype=np.float32).tobytes())
        self.num_docs += 1
        self.num_tokens += len(seq)
        self.max_len = max(self.max_len, len(seq))
        # Flush the small .len/.rmsd streams periodically so a long run's partial
        # output stays inspectable and survives interruption (they otherwise sit
        # in an 8 KiB buffer for thousands of poses).
        if self.num_docs % 256 == 0:
            self.flush()

    def flush(self) -> None:
        """Put every stream on disk.

        Called on a cadence from :meth:`write` and again whenever the meta is
        checkpointed, so the counts in meta.json describe bytes that are
        actually there rather than bytes still in a buffer.
        """
        self._len.flush()
        self._rmsd.flush()
        self._bin.flush()
        self._disp.flush()
        self._dlen.flush()
        self._comp.flush()

    def close(self) -> None:
        self._bin.close()
        self._len.close()
        self._rmsd.close()
        self._disp.close()
        self._dlen.close()
        self._comp.close()


def main() -> None:  # noqa: C901, PLR0915, PLR0912
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--separate-protein-ckpt", type=Path, default=None)
    parser.add_argument("--separate-protein-norm", type=Path, default=None)
    parser.add_argument("--separate-ligand-ckpt", type=Path, default=None)
    parser.add_argument("--separate-ligand-norm", type=Path, default=None)
    parser.add_argument("--norm-stats", type=Path, default=None)
    parser.add_argument("--biolip-dir", type=Path, default=Path("data/biolip"))
    parser.add_argument(
        "--cd-manifest", type=Path, default=Path("data/hub_cache/repo/manifest.parquet")
    )
    parser.add_argument(
        "--casf-pdbs", type=Path, default=Path("data/casf2016_pdbs.txt")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/lm_tokens_decoys"))
    parser.add_argument("--codebook-size", type=int, default=8192)
    parser.add_argument("--max-residues", type=int, default=50)
    parser.add_argument("--n-complexes", type=int, default=12000)
    parser.add_argument(
        "--n-spin-decoys",
        type=int,
        default=0,
        help="extra decoys that only rotate the ligand in place (see _spin)",
    )
    parser.add_argument("--spin-lo-deg", type=float, default=30.0)
    parser.add_argument("--spin-hi-deg", type=float, default=180.0)
    parser.add_argument(
        "--clash-aware",
        action="store_true",
        help="place every decoy so it fits the pocket, as docking would",
    )
    parser.add_argument("--n-decoys", type=int, default=16)
    parser.add_argument("--min-heavy", type=int, default=6)
    parser.add_argument(
        "--max-per-ccd",
        type=int,
        default=0,
        help="cap how many sites one CCD may contribute (0 = no cap)",
    )
    parser.add_argument(
        "--ccd-cap-min-heavy",
        type=int,
        default=0,
        help="only cap CCDs with at least this many heavy atoms",
    )
    parser.add_argument("--max-heavy", type=int, default=50)
    parser.add_argument("--val-frac", type=float, default=0.03)
    parser.add_argument(
        "--site-timeout",
        type=int,
        default=300,
        help="seconds one complex may take before it is abandoned (0 = no "
        "limit). Guards against a single symmetric ligand wedging a shard; see "
        "_SiteBudget for what it does and does not catch.",
    )
    parser.add_argument(
        "--stapled-esm3-cache",
        type=Path,
        default=None,
        help="build the STAPLED BASELINE's corpus instead of ProLIT's: pocket "
        "residues carry ESM3 structure codes read from this cache (pass 2 of "
        "pipelines/corpora/esm3_structure_tokens.py), the ligand carries "
        "ConfSeq tokens, and the placement neither of them holds is sent as "
        "--pose-bits quantized bits. Same sites, same decoys, same labels as "
        "the ProLIT arm -- only the tokenizer differs.",
    )
    parser.add_argument(
        "--confseq-repo", type=Path, default=Path("third_party/ConfSeq")
    )
    parser.add_argument(
        "--stapled-vocab",
        type=Path,
        default=None,
        help="ConfSeq vocabulary JSON: read when emitting, written by "
        "--build-stapled-vocab. Frozen because the ids are baked into every "
        "stream and into the model's embedding table.",
    )
    parser.add_argument(
        "--build-stapled-vocab",
        action="store_true",
        help="collect ConfSeq's chemical symbols over this corpus, write "
        "--stapled-vocab and stop. Run it with --n-decoys 0: the geometric "
        "half of the alphabet is enumerated rather than observed, so only the "
        "symbols need a data pass.",
    )
    parser.add_argument(
        "--dump-receptors",
        type=Path,
        default=None,
        help="pass 1 of the stapled baseline: write the receptor PDB text of "
        "every site this corpus selects to gzipped JSON lines under this "
        "directory and stop. ESM3 has to encode those receptors in another "
        "interpreter, and dumping them from HERE rather than re-deriving the "
        "site list there is what guarantees the two arms are built from the "
        "same complexes -- a second copy of the selection would drift the "
        "moment either side's defaults changed.",
    )
    add_seed_argument(parser, default=0)
    parser.add_argument(
        "--n-near-torsion-decoys",
        type=int,
        default=0,
        help="Of --n-decoys, how many rotate a SINGLE torsion to a target RMSD "
        "in [--near-torsion-lo, --near-torsion-hi], leaving the rest of the "
        "molecule exactly in place. This is the 'right place, one wrong "
        "torsion' class that CASF's decisive 0-1 A band is made of and this "
        "corpus otherwise lacks; see _near_torsion_perturb.",
    )
    parser.add_argument(
        "--mixed-perturb",
        action="store_true",
        help="Sample rigid and torsion magnitudes independently instead of "
        "alternating pure-rigid and torsion-dominated decoys; see "
        "_mixed_perturb for the measurement that motivates it.",
    )
    parser.add_argument("--near-torsion-lo", type=float, default=0.3)
    parser.add_argument("--near-torsion-hi", type=float, default=1.5)
    parser.add_argument(
        "--n-conformer-decoys",
        type=int,
        default=0,
        help="Of --n-decoys, how many come from freshly embedded conformers "
        "superposed on the native pose (docking-like near-natives). The rest "
        "alternate rigid / torsion perturbation as before.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the receptor buckets across this many array tasks.",
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument(
        "--num-rot",
        type=int,
        default=1,
        help="Emit each complex under this many random frame rotations "
        "(1 = canonical frame only). Data augmentation against VQ code noise.",
    )
    parser.add_argument(
        "--skip-canonical",
        action="store_true",
        help="Make every emitted rotation random (no canonical-frame copy), so a "
        "second pass over the same complexes yields rotated duplicates to "
        "concatenate onto an existing canonical corpus.",
    )
    args = parser.parse_args()
    seed_from_args(args)

    from rdkit import RDLogger  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pass 1 for the stapled baseline needs the receptor structures and
    # nothing else: ESM3 cannot run in this interpreter (its package pins a
    # fork of transformers), so the receptors this corpus selects are dumped
    # here and encoded by the benchmark's interpreter. Loading a VQ-VAE and
    # claiming a GPU to copy text would be pure waste.
    stapled = None
    if args.stapled_esm3_cache is not None or args.build_stapled_vocab:
        from prolit.data.esm3_tokens import Esm3TokenCache  # noqa: PLC0415
        from prolit.tokenizers.stapled import ConfSeqVocab  # noqa: PLC0415
        from prolit.tokenizers.stapled_encoder import (  # noqa: PLC0415
            StapledEncoder,
            build_vocab,
        )

        vocab_obj = (
            build_vocab({})
            if args.build_stapled_vocab
            else _load_stapled_vocab(args.stapled_vocab, ConfSeqVocab)
        )
        # The vocabulary pass reads no pocket: ConfSeq's alphabet is a property
        # of the ligands, so it can be collected before ESM3 has encoded a
        # single receptor -- and it has to be, since the corpus cannot be
        # written until the alphabet is frozen.
        cache = (
            Esm3TokenCache(args.stapled_esm3_cache)
            if args.stapled_esm3_cache is not None
            else None
        )
        stapled = StapledEncoder(
            cache=cache,
            confseq_repo=args.confseq_repo,
            vocab=vocab_obj,
            pocket_cfg=PocketExtractionConfig(max_residues=args.max_residues),
        )
        logger.info(
            "stapled baseline: %d receptors cached, vocab %d, "
            "pose %d tokens / %.2f bits",
            len(stapled.cache) if stapled.cache is not None else 0,
            vocab_obj.vocab_size,
            vocab_obj.n_pose_tokens,
            vocab_obj.pose_bits,
        )

    enc = vocab = None
    if not args.dump_receptors and stapled is None:
        cfg = AtomVQVAETrainingConfig()
        cfg.atom.codebook_size = args.codebook_size
        if args.separate_protein_ckpt is not None:
            # ABLATION separate-tokenizers mode: protein-only VQ + ligand-only VQ
            # unified into one code space. Feed RAW descriptors (identity external
            # norm) via PoseEncoder -- SeparateVQVAE normalizes per modality
            # internally. Combined single-range AtomLMVocab over 2*codebook_size
            # codes. PoseEncoder encodes pocket + ligand in separate (single-
            # modality) encode_batch calls, which SeparateVQVAE requires.
            from prolit.tokenizers.descriptor_schema import (  # noqa: PLC0415
                ATOM_DESCRIPTOR_DIM,
            )
            from prolit.tokenizers.separate_vqvae import SeparateVQVAE  # noqa: PLC0415

            module = SeparateVQVAE.from_checkpoints(
                args.separate_protein_ckpt,
                args.separate_protein_norm,
                args.separate_ligand_ckpt,
                args.separate_ligand_norm,
                device,
                codebook_size=args.codebook_size,
            )
            mean = np.zeros(ATOM_DESCRIPTOR_DIM, dtype=np.float32)
            std = np.ones(ATOM_DESCRIPTOR_DIM, dtype=np.float32)
            vocab = AtomLMVocab(codebook_size=2 * args.codebook_size)
        else:
            module = load_atom_vqvae(args.ckpt, device)
            module.eval().to(device)
            norm = torch.load(args.norm_stats, weights_only=False)
            module.vqvae.set_normalization(norm["atom_mean"], norm["atom_std"])
            mean = norm["atom_mean"].numpy()
            std = norm["atom_std"].numpy()
            vocab = AtomLMVocab(codebook_size=args.codebook_size)
        enc = PoseEncoder(
            module.vqvae,
            mean,
            std,
            vocab,
            device,
            PocketExtractionConfig(max_residues=args.max_residues),
        )

    # CASF-excluded, deduped (pdb, ccd) native sites, shuffled to a subset.
    sites = _parse_biolip_txt(args.biolip_dir / "BioLiP.txt.gz")
    ccd_smiles = _load_ccd_smiles(args.biolip_dir / "ligand.tsv.gz")
    excluded = _cd_test_pdbs(args.cd_manifest)
    if args.casf_pdbs.exists():
        excluded |= {p.lower() for p in args.casf_pdbs.read_text().split() if p.strip()}
    rng = np.random.default_rng(args.seed)
    seen: set = set()
    per_ccd: dict[str, int] = {}
    ccd_heavy = (
        _ccd_heavy_atoms(args.biolip_dir / "ligand.tsv.gz")
        if args.max_per_ccd
        else {}
    )
    uniq = []
    for s in sites:
        if s[0] in excluded:
            continue
        key = (s[0], s[2])
        if key in seen:
            continue
        # A handful of cofactors dominate the LARGE end of BioLiP: among sites
        # with >=40 heavy atoms, HEM alone is 20% and the top ten are 57%, so a
        # corpus drawn without a per-ligand cap learns those molecules and
        # little else. Capping what one CCD may contribute triples the
        # peptide-like share (355 -> 1070 complexes in a 20k draw) at the same
        # total size, which is the chemistry CASF's large ligands actually are.
        #
        # That concentration is a property of the large end only. Applying the
        # cap to every ligand thinned the small end instead, and the head that
        # trained on it lost three CASF targets it used to win by 0.6-1.0 A
        # (5c28, 4cr9, 4ih5, all 12-17 heavy atoms) -- 0 gained against 3 lost
        # below 20 atoms, while above 20 it was 9 gained against 2 lost. So cap
        # only the sizes where the concentration exists.
        capped = args.max_per_ccd and (
            ccd_heavy.get(s[2], 0) >= args.ccd_cap_min_heavy
        )
        if capped and per_ccd.get(s[2], 0) >= args.max_per_ccd:
            continue
        per_ccd[s[2]] = per_ccd.get(s[2], 0) + 1
        seen.add(key)
        uniq.append(s)
    rng.shuffle(uniq)
    uniq = uniq[: args.n_complexes]
    logger.info(
        "decoy source: %d native complexes (x%d decoys)", len(uniq), args.n_decoys
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "train": _RmsdWriter(args.out_dir, "train"),
        "val": _RmsdWriter(args.out_dir, "val"),
    }
    symbol_counts: dict[str, int] = {}
    stapled_tally: dict[str, int] = {}
    budget = _SiteBudget(args.site_timeout)
    val_pdbs = {s[0] for s in uniq[: int(len(uniq) * args.val_frac)]}

    # group by bucket to stream each tar once
    by_bucket: dict[str, list[tuple]] = {}
    for s in uniq:
        by_bucket.setdefault(_bucket_code(s[0]), []).append(s)

    from tqdm import tqdm  # noqa: PLC0415

    # Shard by bucket so an array job can build a large corpus in parallel; the
    # complex list and val split are computed identically in every task (same
    # seed), so the shards concatenate into one consistent corpus.
    codes = sorted(by_bucket)[args.shard_id :: args.num_shards]
    logger.info("shard %d/%d: %d buckets", args.shard_id, args.num_shards, len(codes))

    n_ok = 0
    for code in tqdm(codes, desc="buckets"):
        site_list = by_bucket[code]
        needed_rec = {f"{p}{rc}.pdb" for p, rc, _c, _l, _s in site_list}
        needed_lig = {f"{p}_{cc}_{lc}_{s}.pdb" for p, _rc, cc, lc, s in site_list}
        # _read_needed uses a module-global biolip dir; set it once.
        import tokenize_biolip as tb  # noqa: PLC0415

        tb._w_biolip_dir = args.biolip_dir  # noqa: SLF001
        receptors = _read_needed("receptor", code, needed_rec)
        ligands = _read_needed("ligand", code, needed_lig)
        if args.dump_receptors is not None:
            n_ok += _dump_bucket(args.dump_receptors, code, site_list, receptors)
            continue
        for pdb, rchain, ccd, ligchain, serial in site_list:
            rec = receptors.get(f"{pdb}{rchain}.pdb")
            lig = ligands.get(f"{pdb}_{ccd}_{ligchain}_{serial}.pdb")
            if rec is None or lig is None:
                continue
            budget.arm()
            try:
                mol = parse_ligand_pdb_text(
                    lig.decode("utf-8", "replace"), ccd_smiles.get(ccd)
                )
                if mol is None:
                    continue
                heavy_idx = [i for i, a in enumerate(mol["atoms"]) if a[0] != "H"]
                heavy = np.array(
                    [
                        (mol["atoms"][i][1], mol["atoms"][i][2], mol["atoms"][i][3])
                        for i in heavy_idx
                    ],
                    np.float32,
                )
                if not (args.min_heavy <= heavy.shape[0] <= args.max_heavy):
                    continue
                if stapled is not None and args.build_stapled_vocab:
                    spocket = None
                    p_codes = frame = None
                elif stapled is not None:
                    spocket = stapled.setup_pocket(
                        f"{pdb}{rchain}", rec.decode("utf-8", "replace"), heavy
                    )
                    if spocket is None:
                        stapled_tally["pocket_failed"] = (
                            stapled_tally.get("pocket_failed", 0) + 1
                        )
                        continue
                    stapled_tally["sites"] = stapled_tally.get("sites", 0) + 1
                    p_codes = frame = None
                else:
                    spocket = None
                    setup = enc.setup_pocket(rec.decode("utf-8", "replace"), heavy)
                    if setup is None:
                        continue
                    p_codes, frame = setup
                split = "val" if pdb in val_pdbs else "train"
                # Build native + decoys (rigid + conformational), then encode all
                # poses in ONE batched VQ call (per-pose batch-1 was the bottleneck).
                base = np.array([(a[1], a[2], a[3]) for a in mol["atoms"]], np.float64)
                hidx = np.array(heavy_idx)
                mols, rmsds = [mol], [0.0]
                disps = [np.zeros(len(hidx), dtype=np.float32)]
                comps = [(0.0, 0.0)]
                sink = (mols, rmsds, (comps, disps))
                # Every decoy is placed against this: CASF's docking decoys
                # are clash-free at every RMSD band (0.03-0.10 per heavy atom)
                # because a docking program only proposes poses that fit, while
                # a raw perturbation reaches 1.6 at 3-6 A. See :func:`_place`.
                pocket = (
                    _pocket_tree(
                        rec.decode("utf-8", "replace"), base[hidx].mean(0)
                    )
                    if args.clash_aware
                    else None
                )
                # Only the spin class needs this: a 180 deg flip of a
                # symmetric ligand IS the native pose, and labelling it 5 A
                # wrong would teach the head to reject the right answer.
                perms = (
                    _automorphisms(mol["atoms"], mol["bonds"], hidx)
                    if args.n_spin_decoys > 0
                    else None
                )
                # Freshly embedded conformers superposed on the native: the
                # docking-like near-native class that perturbation cannot make.
                confs = (
                    _conformer_decoys(
                        mol["atoms"],
                        mol["bonds"],
                        hidx,
                        base,
                        args.n_conformer_decoys,
                        # crc32, not hash(): PYTHONHASHSEED randomizes str hashes
                        # per process, so shards would not be reproducible.
                        seed=zlib.crc32(f"{pdb}_{ccd}".encode()) % (2**31),
                    )
                    if args.n_conformer_decoys > 0
                    else []
                )
                n_near = args.n_near_torsion_decoys
                n_pert = args.n_decoys - len(confs) - n_near
                for k in range(args.n_decoys):
                    new = _placed(
                        _decoy_drawer(
                            k, n_near, n_pert, confs, mol, base, hidx, rng, args
                        ),
                        hidx,
                        pocket,
                        rng,
                    )
                    if new is not None:
                        _record_decoy(new, mol, base, hidx, sink)
                spin = functools.partial(
                    _spin, base, hidx, rng, args.spin_lo_deg, args.spin_hi_deg
                )
                for _ in range(args.n_spin_decoys):
                    spun = _placed(spin, hidx, pocket, rng)
                    if spun is not None:
                        _record_decoy(
                            spun, mol, base, hidx, sink, perms=perms
                        )
                if stapled is not None:
                    n_ok += _emit_stapled(
                        stapled, spocket, mols, rmsds, comps, hidx,
                        writers[split], symbol_counts, stapled_tally,
                        build_vocab_only=args.build_stapled_vocab,
                    )
                    continue
                # Descriptors once; each extra rotation only re-quantizes them.
                # A rotated frame is a different tokenization of the SAME complex
                # (the VQ-VAE was pretrained with this augmentation), so it
                # teaches the head to score the pose, not the code pattern.
                descs = enc.ligand_descs(mols, frame)
                for r in range(args.num_rot):
                    canon = r == 0 and not args.skip_canonical
                    rot = None if canon else random_rotation_matrix(rng)
                    pc = p_codes if rot is None else enc.pocket_codes_rotated(rot)
                    seqs = enc.seqs_from_descs(pc, descs, rotation=rot)
                    if seqs[0] is None:
                        break  # native must encode
                    for seq, rmsd, dsp, cmp_, dsc in zip(
                        seqs, rmsds, disps, comps, descs, strict=True
                    ):
                        if seq is None:
                            continue
                        # Per-atom labels are only usable when they line up with
                        # the emitted ligand tokens (one row per heavy atom).
                        ok = dsc.shape[0] == dsp.shape[0]
                        writers[split].write(
                            seq, rmsd, dsp if ok else None, cmp_
                        )
                else:
                    n_ok += 1
            except TimeoutError:
                logger.warning(
                    "site %s_%s exceeded %ds; skipped", pdb, ccd, budget.seconds
                )
                continue
            except Exception:
                logger.exception("failed %s_%s", pdb, ccd)
                continue
            finally:
                budget.disarm()
        # One bucket done: put a meta.json on disk describing exactly what has
        # been written. Costs ~700 bytes of JSON per bucket and means a shard
        # that wedges, or that a walltime kills, still hands its finished
        # complexes to concat_decoy_shards.py instead of taking them with it.
        if args.dump_receptors is None and not args.build_stapled_vocab:
            _write_meta(
                args.out_dir, args, vocab, stapled, n_ok, stapled_tally, writers
            )

    if budget.hit:
        logger.warning("%d complexes abandoned on the time budget", budget.hit)
    if stapled is not None:
        logger.info("stapled coverage: %s", stapled_tally)
    if args.build_stapled_vocab:
        from prolit.tokenizers.stapled_encoder import build_vocab  # noqa: PLC0415

        built = build_vocab(symbol_counts)
        args.stapled_vocab.parent.mkdir(parents=True, exist_ok=True)
        built.confseq.save(args.stapled_vocab)
        logger.info(
            "stapled vocabulary: %d ConfSeq tokens (%d observed) -> %s",
            built.confseq.size,
            len(symbol_counts),
            args.stapled_vocab,
        )
        for w in writers.values():
            w.close()
        return

    if args.dump_receptors is not None:
        logger.info("dumped %d receptors -> %s", n_ok, args.dump_receptors)
        for w in writers.values():
            w.close()
        return

    # Meta first: it flushes the writers, so closing after it adds nothing and
    # the file on disk describes exactly what was written.
    _write_meta(args.out_dir, args, vocab, stapled, n_ok, stapled_tally, writers)
    for split, w in writers.items():
        w.close()
        logger.info("%s: %d docs (%d tokens)", split, w.num_docs, w.num_tokens)
    logger.info("wrote decoy token set to %s", args.out_dir)


if __name__ == "__main__":
    main()

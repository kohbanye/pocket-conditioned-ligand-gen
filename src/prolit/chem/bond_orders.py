"""Turn decoded per-atom chemistry plus coordinates back into a molecule.

Reconstruction can carry the reference bond graph through (see the recon
bench), but generation cannot: nothing upstream knows whether a given ring is
benzene or cyclohexane. The obvious fix -- give bond order its own token -- was
measured to be unnecessary. The descriptor already predicts ``numH``, and for
atom *i* the orders of its bonds to heavy neighbours must sum to

    valence(element, charge) - numH(i)

so the excess over an all-single graph, ``e_i = target_i - degree_i``, is
handed out along the bonds. Promoting one bond to a double raises both of its
endpoints by 1, so the promotions form a subgraph in which atom *i* has degree
``e_i`` -- a b-matching, solved exactly. A benzene ring falls out as a Kekule
structure by construction, and RDKit's own aromaticity perception then labels
it during sanitisation, so "aromatic vs double" never has to be decided here.

Measured on 1074 CrossDocked test ligands (fold 0 test side, CASF and the
sbdd-bench targets held out) given the true heavy-atom graph and the reference
per-atom chemistry: every one comes back with the reference SMILES, 1074/1074.
Bond order therefore costs no tokens.

A further 59 ligands were left out of that count because the descriptor's own
chemistry cannot express them at all -- 33-D per-atom features lose something
the reference molecule has, independently of this solver -- so the honest
reading is that 94.8% of ligands are expressible and 100% of those are
recovered. The residual is a ceiling of the descriptor, not of this module.

RDKit's ``rdDetermineBonds.DetermineBondOrders`` solves the same problem but
requires every atom including hydrogen, which a heavy-atom decoder does not
have; that is why this exists rather than a call into RDKit.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rdkit.Chem import Mol

# Neutral heavy-atom valences. Sulfur and phosphorus have several -- a
# sulfonamide S is 6-valent where a thioether S is 2-valent -- and nothing in
# the per-atom descriptor distinguishes them, so they are tried in order.
_VALENCE: dict[str, tuple[int, ...]] = {
    "C": (4,),
    "N": (3,),
    "O": (2,),
    "S": (2, 4, 6),
    "P": (3, 5),
    "F": (1,),
    "Cl": (1,),
    "Br": (1,),
    "I": (1,),
    "B": (3,),
    "Si": (4,),
}

#: Elements whose valence a formal charge raises (N+ is 4-valent, O- is
#: 1-valent). For carbon either sign removes a bond, hence the ``abs``.
_CHARGE_ADDS = frozenset({"N", "P", "O", "S"})

#: Ring sizes that can carry an aromatic sextet in what the decoder emits.
_AROMATIC_RING_SIZES = (5, 6)

#: A single atom is trivially one piece, so nothing to join.
_MIN_ATOMS_TO_CONNECT = 2

#: Cap on the valence combinations tried for the multi-valent elements. 27 is
#: three sulfurs' worth; beyond that the molecule is pathological and a miss is
#: cheaper than the search.
_MAX_VALENCE_COMBOS = 27


def target_bond_sums(element: str, charge: int, num_h: int) -> tuple[int, ...]:
    """Candidate sums of bond orders to heavy neighbours, most common first.

    Empty when the element has no tabulated valence, which is the caller's
    signal that the molecule cannot be built.
    """
    valences = _VALENCE.get(element)
    if valences is None:
        return ()
    out = []
    for valence in valences:
        if element in _CHARGE_ADDS:
            adjusted = valence + charge
        elif element == "C":
            adjusted = valence - abs(charge)
        else:
            adjusted = valence
        out.append(adjusted - num_h)
    return tuple(out)


def _is_bridge(bonds: list[tuple[int, int]], bond: tuple[int, int]) -> bool:
    """Would removing ``bond`` disconnect the two atoms it joins?"""
    adjacency: dict[int, list[int]] = {}
    for i, j in bonds:
        if (i, j) == bond:
            continue
        adjacency.setdefault(i, []).append(j)
        adjacency.setdefault(j, []).append(i)
    start, goal = bond
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        if node == goal:
            return False
        for neighbour in adjacency.get(node, ()):
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return True


def connect_fragments(
    elements: Sequence[str],
    bonds: Sequence[tuple[int, int]],
    coords: np.ndarray,
) -> list[tuple[int, int]]:
    """Join a perceived graph that came back in pieces, shortest gap first.

    :func:`prune_to_valence` fixes perception being too generous. This fixes
    the mirror case: a fixed distance tolerance is also sometimes too *tight*,
    and a molecule the model meant as one piece arrives as two. It then fails
    ``all_atoms_connected`` outright -- the largest single PoseBusters failure
    left once the valence pruning is in place, at 0.119 of molecules.

    The repair carries no threshold of its own. Perception already ranked every
    candidate pair by separation against the summed covalent radii; this walks
    that ranking upward and takes the shortest link between two components,
    repeatedly, until one component is left. Which is to say the tolerance
    becomes per molecule: exactly wide enough for this molecule to be whole,
    and no wider.

    Run it **before** :func:`prune_to_valence`, not after: the links added here
    are bridges by construction, and the pruning refuses to cut a bridge, so
    the two repairs compose rather than fight. Joining afterwards would instead
    let a new link over-coordinate an atom the pruning had just fixed.
    """
    symbols = [str(e) for e in elements]
    n = len(symbols)
    kept = [(int(i), int(j)) for i, j in bonds]
    if n < _MIN_ATOMS_TO_CONNECT:
        return kept

    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in kept:
        parent[find(i)] = find(j)
    for _, i, j in _ranked_pairs(symbols, coords):
        if len({find(k) for k in range(n)}) == 1:
            break
        if find(i) == find(j):
            continue
        kept.append((i, j))
        parent[find(i)] = find(j)
    return kept


def _ranked_pairs(
    symbols: list[str], coords: np.ndarray
) -> list[tuple[float, int, int]]:
    """Every atom pair, shortest first, measured against its summed radii."""
    from prolit.chem.pdb_io import covalent_radius  # noqa: PLC0415

    out = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            reference = covalent_radius(symbols[i]) + covalent_radius(symbols[j])
            if reference > 0.0:
                out.append(
                    (float(np.linalg.norm(coords[i] - coords[j])) / reference, i, j)
                )
    out.sort()
    return out


def relax_numh_for_aromatic_atoms(
    elements: Sequence[str],
    charges: Sequence[int],
    num_h: Sequence[int],
    bonds: Sequence[tuple[int, int]],
    aromatic: Sequence[bool],
) -> list[int]:
    """Give the atoms the model called aromatic the bond order to be aromatic.

    An aromatic CH needs its bond orders to sum to three. The decoder predicts
    ``numH`` with a classifier, and one hydrogen too many takes that sum to
    two -- at which point the only feasible assignment is all-single and the
    ring comes out saturated. That is not hypothetical: before this, the median
    generated molecule had **fsp3 = 1.00** (every carbon sp3) against the
    reference ligands' 0.50, and 0.76 aromatic rings against 1.31.

    The decoder already answers the question. ``aromatic`` is one of its
    per-atom heads, decoded on the way out and -- until this existed -- used
    only as a node feature for the pose refiner, never to build the molecule.
    So where the model says aromatic and ``numH`` leaves no room for a pi bond,
    the hydrogens go and the room appears. Nothing is inferred and no threshold
    is introduced: it is the model's own prediction, applied to the molecule
    the model is describing.

    Taking back exactly one hydrogen is not enough. A substituted ring carbon
    (three neighbours, ``numH`` 2) goes from a sum of two to three and is still
    forced all-single, so the deficit is computed against the neighbour count.
    """
    symbols = [str(e) for e in elements]
    adjusted = [int(h) for h in num_h]
    degree: dict[int, int] = {}
    for i, j in bonds:
        degree[i] = degree.get(i, 0) + 1
        degree[j] = degree.get(j, 0) + 1

    for i, is_aromatic in enumerate(aromatic):
        if not is_aromatic or adjusted[i] <= 0:
            continue
        sums = target_bond_sums(symbols[i], int(charges[i]), adjusted[i])
        if not sums:
            continue
        want = degree.get(i, 0) + 1 - max(sums)
        if want > 0:
            adjusted[i] -= min(want, adjusted[i])
    return adjusted


def prune_to_valence(
    elements: Sequence[str],
    charges: Sequence[int],
    num_h: Sequence[int],
    bonds: Sequence[tuple[int, int]],
    coords: np.ndarray,
) -> list[tuple[int, int]]:
    """Drop perceived bonds that over-coordinate an atom, weakest first.

    Distance perception has no idea what a valence is, so two decoded atoms
    that land too close acquire a bond, and the atom that already had four
    neighbours acquires a fifth. :func:`assign_bond_orders` then returns
    ``None`` -- no promotion can undo an atom that is already over-coordinated
    -- and the molecule falls out of the chemistry-aware path entirely: it
    cannot be sanitised, so it cannot be relaxed, so its bond lengths and
    angles stay as the decoder left them.

    That chain was measured. Of generated ligands, 18.2% fail to sanitise, and
    they carry PB-validity 0.543 against 0.805 for the rest; their shortest
    bond sits at 0.687 of its reference length while the relaxed molecules'
    sits at 0.949. Loosening the relaxation restraint from 0.102 A to 1.0 A
    moved the shortest-bond ratio by 0.001, because the molecules that need it
    are exactly the ones the force field will not type.

    So rather than give up, this drops bonds until every atom fits its
    valence. Which bond to drop is not a choice: perception ranked them by
    length against the summed covalent radii, and the longest such ratio is
    the one it was least sure of -- but only among the cuts that keep the
    molecule in one piece. Cutting a bridge splits it, and a split molecule
    fails ``all_atoms_connected``; taking the longest bond without that check
    took the sanitisable fraction to 1.000 and the connected check from 0.070
    to 0.215 failing, trading one PoseBusters failure for another. So the
    order is: keep it connected, then cut the doubtful bond. Where every
    remaining candidate is a bridge this gives up and returns the graph
    unchanged, which is exactly where the caller stands without it -- the
    pruning can rescue a molecule but never costs one.

    **Only for perceived graphs.** Where the real bond graph is known
    (reconstruction), an over-coordinated atom means the input said so, and
    deleting the evidence would be the wrong repair.
    """
    from prolit.chem.pdb_io import covalent_radius  # noqa: PLC0415

    symbols = [str(e) for e in elements]
    limit: dict[int, int] = {}
    for i, symbol in enumerate(symbols):
        sums = target_bond_sums(symbol, int(charges[i]), int(num_h[i]))
        # The largest tabulated option is the most room the atom could have;
        # anything beyond it is over-coordinated whichever valence it turns out
        # to take.
        limit[i] = max(sums) if sums else 0

    kept = [(int(i), int(j)) for i, j in bonds]

    def _ratio(i: int, j: int) -> float:
        """Bond length over what the two elements' covalent radii expect.

        ``covalent_radius`` returns 0.0 for an element it does not tabulate, and
        the decoder can emit those -- ``LIGAND_ELEMENT_VOCAB`` has an ``OTHER``
        slot. Distance perception never proposes such a bond (it masks on a
        positive radius) so this used to be unreachable; a learned bond graph
        does propose them, and the division took down the whole molecule. With
        no expectation to measure against, the bond is maximally doubtful, so
        it is the first one cut.
        """
        scale = covalent_radius(symbols[i]) + covalent_radius(symbols[j])
        if scale <= 0.0:
            return float("inf")
        return float(np.linalg.norm(coords[i] - coords[j])) / scale

    ratio = {(i, j): _ratio(i, j) for i, j in kept}
    while True:
        degree: dict[int, int] = {}
        for i, j in kept:
            degree[i] = degree.get(i, 0) + 1
            degree[j] = degree.get(j, 0) + 1
        over = [i for i, d in degree.items() if d > limit.get(i, 0)]
        if not over:
            return kept
        over_set = set(over)
        candidates = [b for b in kept if b[0] in over_set or b[1] in over_set]
        if not candidates:
            return kept
        safe = [b for b in candidates if not _is_bridge(kept, b)]
        if not safe:
            # Every remaining candidate is a bridge -- cutting any of them
            # splits the molecule, and the pieces fail both the solver (an
            # atom left short of its bond sum) and ``all_atoms_connected``.
            # Give up instead, which leaves the caller exactly where it is
            # without this.
            return kept
        kept.remove(max(safe, key=lambda b: ratio[b]))


def assign_bond_orders(
    elements: Sequence[str],
    charges: Sequence[int],
    num_h: Sequence[int],
    bonds: Sequence[tuple[int, int]],
) -> dict[tuple[int, int], int] | None:
    """Map each bond to an order, or ``None`` when no assignment is consistent.

    ``None`` is a real answer, not a fallback: it means the decoded chemistry
    and the perceived connectivity contradict each other, and filling in single
    bonds anyway would hide that behind a molecule nobody asked for.
    """
    candidates = [
        target_bond_sums(str(e), int(charges[i]), int(num_h[i]))
        for i, e in enumerate(elements)
    ]
    if any(not c for c in candidates):
        return None
    ambiguous = [i for i, c in enumerate(candidates) if len(c) > 1]
    settled = {i: c[0] for i, c in enumerate(candidates) if len(c) == 1}
    combos = list(
        itertools.islice(
            itertools.product(*(candidates[i] for i in ambiguous)),
            _MAX_VALENCE_COMBOS,
        )
    )
    for combo in combos or [()]:
        targets = dict(settled)
        targets.update(dict(zip(ambiguous, combo, strict=True)))
        order = _solve(bonds, targets)
        if order is not None:
            return order
    return None


def _excess_valence(
    bonds: Sequence[tuple[int, int]], targets: dict[int, int]
) -> dict[int, int] | None:
    """Bond orders each atom still owes beyond an all-single graph.

    ``None`` when an atom already has more neighbours than its valence allows,
    which no promotion can undo.
    """
    degree: dict[int, int] = dict.fromkeys(targets, 0)
    for i, j in bonds:
        degree[i] = degree.get(i, 0) + 1
        degree[j] = degree.get(j, 0) + 1
    excess = {}
    for i, target in targets.items():
        left = target - degree.get(i, 0)
        if left < 0:
            return None
        excess[i] = left
    return excess


def _solve(
    bonds: Sequence[tuple[int, int]], targets: dict[int, int]
) -> dict[tuple[int, int], int] | None:
    """Exact b-matching for one fixed choice of valences."""
    import networkx as nx  # noqa: PLC0415

    excess = _excess_valence(bonds, targets)
    if excess is None:
        return None
    single: dict[tuple[int, int], int] = {
        (min(a, b), max(a, b)): 1 for a, b in bonds
    }
    if not any(excess.values()):
        return single

    # b-matching by node splitting: an atom needing ``e`` extra bond orders
    # becomes ``e`` copies, and a perfect matching on the expanded graph is
    # exactly a valid set of promotions.
    split = nx.Graph()
    for i, left in excess.items():
        split.add_nodes_from((i, k) for k in range(left))
    for a, b in bonds:
        split.add_edges_from(
            ((a, ka), (b, kb))
            for ka in range(excess.get(a, 0))
            for kb in range(excess.get(b, 0))
        )
    matching = nx.max_weight_matching(split, maxcardinality=True)
    if 2 * len(matching) != split.number_of_nodes():
        return None  # no perfect matching -> the valences cannot all be met
    order = dict(single)
    for (a, _), (b, _) in matching:
        order[min(a, b), max(a, b)] += 1
    return order


def mol_from_decoded(  # noqa: PLR0913
    elements: Sequence[str],
    charges: Sequence[int],
    num_h: Sequence[int],
    coords: np.ndarray,
    bonds: Sequence[tuple[int, int]] | None = None,
    *,
    perceived: bool | None = None,
    aromatic: Sequence[bool] | None = None,
) -> Mol | None:
    """Decoded atoms + coordinates -> a sanitised RDKit molecule, or ``None``.

    ``bonds`` defaults to distance perception over ``coords``
    (:func:`~prolit.chem.pdb_io.infer_bonds`). Pass the real graph when one is
    known -- in reconstruction it is, and perceiving it again only adds a
    second thing that can be wrong.

    ``perceived`` says whether that graph came from the coordinates rather than
    from a known molecule; it defaults to ``bonds is None``. ``aromatic`` is the
    decoder's per-atom aromaticity head. Together they earn
    :func:`relax_numh_for_aromatic_atoms`: where the graph was perceived and the
    model called an atom aromatic, a ``numH`` that leaves no room for a pi bond
    is overruled. A known graph carries its own chemistry and is left alone.

    Hydrogens are set explicitly from the decoded ``numH`` rather than left to
    RDKit: the count is the very quantity the bond orders were derived from, so
    letting RDKit re-infer it would let sanitisation quietly disagree with the
    solver.
    """
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Geometry import Point3D  # noqa: PLC0415

    from prolit.chem.pdb_io import infer_bonds  # noqa: PLC0415

    symbols = [str(e) for e in elements]
    if perceived is None:
        perceived = bonds is None
    if bonds is None:
        bonds = infer_bonds(symbols, coords)
    counts = list(num_h)
    if perceived and aromatic is not None:
        # Only for a perceived graph: where the real bonds are known the numH
        # that came with them is known too, and second-guessing it would be
        # inventing chemistry rather than recovering it.
        counts = relax_numh_for_aromatic_atoms(
            symbols, charges, counts, bonds, aromatic
        )
    order = assign_bond_orders(symbols, charges, counts, bonds)
    if order is None and counts != list(num_h):
        counts = list(num_h)
        order = assign_bond_orders(symbols, charges, counts, bonds)
    if order is None:
        return None
    num_h = counts

    rw = Chem.RWMol()
    for i, symbol in enumerate(symbols):
        atom = Chem.Atom(symbol)
        atom.SetFormalCharge(int(charges[i]))
        # RDKit's setters take the flag positionally; there is no keyword form.
        atom.SetNoImplicit(True)  # noqa: FBT003
        atom.SetNumExplicitHs(int(num_h[i]))
        rw.AddAtom(atom)
    kinds = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    for (i, j), value in order.items():
        rw.AddBond(int(i), int(j), kinds.get(int(value), Chem.BondType.SINGLE))

    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(idx, Point3D(float(x), float(y), float(z)))
    # Without this the conformer carries RDKit's default 2D flag, and every
    # writer stamps the mol block "2D" while filling in a z column. Readers
    # then warn and override it, so nothing breaks today -- but a consumer that
    # believes the flag would flatten the pose, which is the kind of silent
    # wrongness this codebase pays for elsewhere.
    conf.Set3D(True)  # noqa: FBT003
    mol.AddConformer(conf, assignId=True)
    try:
        Chem.SanitizeMol(mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError):
        return None
    return mol

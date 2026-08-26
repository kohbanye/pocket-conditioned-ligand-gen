"""The bond head must respect valence and use what distance alone cannot.

The head exists because distance perception recovers 31% of the bond graph at
the error the decoder makes. Two properties have to hold for its output to be
usable at all: no atom may be given more bonds than its own predicted chemistry
allows, and the features it is handed must actually carry the chemistry rather
than being a repackaged distance.
"""

from __future__ import annotations

import numpy as np
import torch

from prolit.chem.bond_orders import prune_to_valence
from prolit.model.bond_head import (
    MAX_GAP,
    BondHead,
    bond_capacity,
    bond_jaccard,
    bonds_from_head,
    pair_features,
)
from prolit.model.pose_refiner import FEATURE_FIELDS, ligand_feats_from_heads
from prolit.tokenizers.descriptor_schema import LIGAND_ELEMENT_VOCAB

NAMES = [n for n, _ in FEATURE_FIELDS]


def _feats(elements: list[int], num_h: list[int]) -> np.ndarray:
    n = len(elements)
    return ligand_feats_from_heads(
        {
            "element": np.array(elements),
            "charge": np.full(n, 2),  # index of charge 0
            "hybrid": np.zeros(n, dtype=int),
            "aromatic": np.zeros(n, dtype=int),
            "ring": np.zeros(n, dtype=int),
            "numH": np.array(num_h),
        },
        n,
    ).astype(np.int64)


def test_capacity_falls_as_the_atom_carries_more_hydrogens() -> None:
    # Carbon with 0, 1, 2, 3 hydrogens: 4, 3, 2, 1 bond orders left to spend.
    feats = _feats([0, 0, 0, 0], [0, 1, 2, 3])
    assert list(bond_capacity(feats)) == [4.0, 3.0, 2.0, 1.0]


def test_pair_features_cover_every_pair_and_carry_the_token_gap() -> None:
    n = 5
    coords = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    feats = _feats([0] * n, [0] * n)
    cont, cat, i, j = pair_features(coords, feats)
    assert len(cont) == len(cat) == n * (n - 1) // 2
    assert (j > i).all()
    # The last categorical column is the clipped token-order gap.
    assert (cat[:, -1] == np.minimum(j - i, MAX_GAP)).all()


def test_no_atom_is_given_more_bonds_than_its_valence_allows() -> None:
    torch.manual_seed(0)
    head = BondHead(dim=4, hidden=16)
    # Bias the head to say "bonded" everywhere; the budget is the only brake.
    with torch.no_grad():
        head.net[-1].bias.fill_(20.0)
    n = 8
    coords = np.random.default_rng(0).normal(0, 1.5, (n, 3)).astype(np.float32)
    # Every atom is a carbon carrying three hydrogens: one bond each, at most.
    feats = _feats([0] * n, [3] * n)
    bonds = bonds_from_head(head, coords, feats)
    degree = np.zeros(n)
    for a, b in bonds:
        degree[a] += 1
        degree[b] += 1
    assert degree.max() <= 1.0
    assert all(a != b for a, b in bonds)


def test_jaccard_is_over_unordered_pairs() -> None:
    assert bond_jaccard([(1, 0)], [(0, 1)]) == 1.0
    assert bond_jaccard([(0, 1), (1, 2)], [(0, 1)]) == 0.5
    assert bond_jaccard([], []) == 0.0


def test_a_two_atom_molecule_and_a_lone_atom_do_not_crash() -> None:
    head = BondHead(dim=4, hidden=16)
    lone = np.zeros((1, 3), dtype=np.float32)
    assert bonds_from_head(head, lone, _feats([0], [0])) == []
    coords = np.array([[0.0, 0, 0], [1.5, 0, 0]], dtype=np.float32)
    out = bonds_from_head(head, coords, _feats([0, 0], [0, 0]))
    assert out in ([], [(0, 1)])


def test_an_untabulated_element_is_never_given_a_bond() -> None:
    """``LIGAND_ELEMENT_VOCAB`` has an OTHER slot and the decoder emits it.

    Distance perception skips those atoms, so nothing downstream has ever seen
    a bond to one -- ``prune_to_valence`` divided by their (zero) covalent
    radius and took down the molecule. Seven of a hundred targets generated
    nothing at all that way.
    """
    torch.manual_seed(0)
    head = BondHead(dim=4, hidden=16)
    with torch.no_grad():
        head.net[-1].bias.fill_(20.0)  # say "bonded" to everything
    other = len(LIGAND_ELEMENT_VOCAB) - 1
    assert LIGAND_ELEMENT_VOCAB[other] == "OTHER"
    feats = _feats([0, 0, other], [0, 0, 0])
    coords = np.array([[0.0, 0, 0], [1.5, 0, 0], [3.0, 0, 0]], dtype=np.float32)
    bonds = bonds_from_head(head, coords, feats)
    assert all(2 not in pair for pair in bonds)


def test_prune_to_valence_survives_a_bond_to_an_untabulated_element() -> None:
    coords = np.array([[0.0, 0, 0], [1.5, 0, 0], [3.0, 0, 0]], dtype=np.float64)
    kept = prune_to_valence(
        ["C", "C", "X"], [0, 0, 0], [0, 0, 0], [(0, 1), (1, 2)], coords
    )
    # The C-C bond survives; the one to the untabulated atom is the doubtful one.
    assert (0, 1) in kept

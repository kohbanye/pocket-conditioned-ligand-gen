"""Read the bond graph off the decoded atoms instead of off their distances.

The generation path perceives bonds from the decoded coordinates
(``infer_bonds``: bonded iff closer than the summed covalent radii plus a
tolerance). That works when the coordinates are right. They are not: the
language model's decode sits 2.5 A from the crystal pose, and at that error
the two distributions overlap past saving --

    perceived from            Jaccard vs the true bonds
    crystal coordinates            1.000
    LM-decoded coordinates         0.309   (44% of bonds missed, 63% invented)

and no threshold fixes it: catching 55% of the true bonds already invents 0.56
false ones per true one, and reaching 78% invents 2.4. Bonding consecutive
atoms in token order and ignoring the coordinates entirely scores *higher*
(0.377) than the deployed perception.

That matters far past the bond list. Molecule identity is read off the graph,
so a broken graph is a broken molecule: with the chemistry heads held fixed and
only the coordinates perturbed, a 0.43 A displacement takes the median
aromatic-ring count from 2 to 0 and the fraction that sanitise at all from 0.71
to 0.06. The generated molecules' aromatic deficit, their fsp3 of 0.94 and
their QED of 0.35 are all this.

But the codes carry more than a position. Each one decodes to an element, a
charge, a hybridisation, an aromatic flag, a ring flag and a hydrogen count --
and the token order is known at generation time. This head reads a bond
probability off all of it, and is trained on exactly the input it will meet:
the pose-refine corpus holds the LM's own decoded coordinates beside the
crystal molecule's true bonds. Measured on that corpus's validation split,
Jaccard 0.309 -> 0.720 and the fraction of molecules whose connectivity is
*exactly* right, 0.000 -> 0.162.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn

from prolit.tokenizers.descriptor_schema import (
    LIGAND_CHARGE_VOCAB,
    LIGAND_ELEMENT_VOCAB,
    LIGAND_NUMH_VOCAB,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Token-order distance between two atoms, clipped. Consecutive atoms carry 52%
#: of all bonds and |i-j| <= 2 carries 67%, so the gap is a feature in its own
#: right; past this the prior is flat and the bucket is a catch-all.
MAX_GAP = 12

#: Categorical per-atom fields the head reads, in the order it embeds them.
#: These are columns of the pose-refiner node-feature block, so training and
#: generation hand the head the same thing.
CHEM_FIELDS: tuple[str, ...] = (
    "element",
    "charge",
    "hybrid",
    "aromatic",
    "ring",
    "numH",
)

#: Continuous pair features. Distance is here twice, in raw and squashed form,
#: because the decision is sharp near contact and flat far from it.
NUM_CONTINUOUS = 7


def _chem_columns() -> list[int]:
    from prolit.model.pose_refiner import FEATURE_FIELDS  # noqa: PLC0415

    names = [n for n, _ in FEATURE_FIELDS]
    return [names.index(f) for f in CHEM_FIELDS]


def _chem_sizes() -> list[int]:
    from prolit.model.pose_refiner import FEATURE_FIELDS  # noqa: PLC0415

    sizes = dict(FEATURE_FIELDS)
    return [sizes[f] for f in CHEM_FIELDS]


def bond_capacity(
    feats: np.ndarray, columns: list[int] | None = None
) -> np.ndarray:
    """Largest total bond order each atom could carry, from its own chem heads.

    An atom that already has all the bonds its valence allows cannot take
    another one, and how much room is left changes a pair's odds far more than
    its distance does -- so the head is given the budget rather than left to
    infer it from the element.
    """
    from prolit.chem.bond_orders import target_bond_sums  # noqa: PLC0415

    cols = columns or _chem_columns()
    e_col, c_col, h_col = cols[0], cols[1], cols[5]
    out = np.zeros(len(feats), dtype=np.float32)
    for i, row in enumerate(feats):
        element = LIGAND_ELEMENT_VOCAB[int(row[e_col])]
        if element == "OTHER":
            element = "C"
        sums = target_bond_sums(
            element,
            LIGAND_CHARGE_VOCAB[int(row[c_col])],
            LIGAND_NUMH_VOCAB[int(row[h_col])],
        )
        out[i] = max(sums) if sums else 0.0
    return out


def pair_features(
    coords: np.ndarray, feats: np.ndarray, columns: list[int] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every atom pair's (continuous, categorical) features, plus the pair index.

    Returns ``(continuous, categorical, i, j)`` over the upper triangle.
    """
    from prolit.chem.pdb_io import covalent_radius  # noqa: PLC0415

    cols = columns or _chem_columns()
    n = len(coords)
    coords = np.asarray(coords, dtype=np.float32)
    radii = np.array(
        [
            covalent_radius(
                LIGAND_ELEMENT_VOCAB[int(r[cols[0]])]
                if LIGAND_ELEMENT_VOCAB[int(r[cols[0]])] != "OTHER"
                else "C"
            )
            for r in feats
        ],
        dtype=np.float32,
    )
    cap = bond_capacity(feats, cols)
    i, j = np.triu_indices(n, 1)
    d = np.linalg.norm(coords[i] - coords[j], axis=-1)
    excess = d - (radii[i] + radii[j])
    gap = np.minimum(j - i, MAX_GAP)
    continuous = np.stack(
        [
            d,
            excess,
            np.exp(-np.clip(excess, 0.0, None)),
            1.0 / (1.0 + d),
            cap[i],
            cap[j],
            np.minimum(cap[i], cap[j]),
        ],
        axis=-1,
    ).astype(np.float32)
    categorical = np.concatenate(
        [feats[i][:, cols], feats[j][:, cols], gap[:, None]], axis=-1
    ).astype(np.int64)
    return continuous, categorical, i, j


class BondHead(nn.Module):
    """Pair classifier: decoded chemistry + geometry -> P(bonded)."""

    def __init__(self, dim: int = 48, hidden: int = 256) -> None:
        super().__init__()
        sizes = [*_chem_sizes(), *_chem_sizes(), MAX_GAP + 1]
        self.embeddings = nn.ModuleList(nn.Embedding(s, dim) for s in sizes)
        self.net = nn.Sequential(
            nn.Linear(NUM_CONTINUOUS + dim * len(sizes), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, continuous: Tensor, categorical: Tensor) -> Tensor:
        embedded = torch.cat(
            [m(categorical[:, k]) for k, m in enumerate(self.embeddings)], dim=-1
        )
        return self.net(torch.cat([continuous, embedded], dim=-1)).squeeze(-1)


@torch.no_grad()
def bonds_from_head(
    head: nn.Module,
    coords: np.ndarray,
    feats: np.ndarray,
    *,
    device: torch.device | None = None,
    threshold: float = 0.5,
) -> list[tuple[int, int]]:
    """Assemble a bond graph from the head's probabilities.

    Confident pairs first, and a pair is only taken while both atoms still have
    valence to spend -- the head scores pairs independently, and the budget is
    the one constraint that couples them.
    """
    device = device or next(head.parameters()).device
    n = len(coords)
    if n < 2:  # noqa: PLR2004
        return []
    cols = _chem_columns()
    continuous, categorical, idx_i, idx_j = pair_features(coords, feats, cols)
    prob = (
        torch.sigmoid(
            head(
                torch.from_numpy(continuous).to(device),
                torch.from_numpy(categorical).to(device),
            )
        )
        .cpu()
        .numpy()
    )
    cap = bond_capacity(feats, cols) * _bondable(feats, cols)
    degree = np.zeros(n, dtype=np.float32)
    bonds: list[tuple[int, int]] = []
    for k in np.argsort(-prob):
        if prob[k] < threshold:
            break
        a, b = int(idx_i[k]), int(idx_j[k])
        if degree[a] < cap[a] and degree[b] < cap[b]:
            bonds.append((a, b))
            degree[a] += 1
            degree[b] += 1
    return bonds


def _bondable(feats: np.ndarray, columns: list[int]) -> np.ndarray:
    """Atoms whose element has a covalent radius, and so can carry a bond.

    ``LIGAND_ELEMENT_VOCAB`` has an ``OTHER`` slot and the decoder does emit it.
    Distance perception silently skips those atoms; the head has to skip them
    too, or the graph it proposes is one the rest of the pipeline has never
    been asked to handle.
    """
    from prolit.chem.pdb_io import covalent_radius  # noqa: PLC0415

    return np.array(
        [
            covalent_radius(
                LIGAND_ELEMENT_VOCAB[int(r[columns[0]])]
                if LIGAND_ELEMENT_VOCAB[int(r[columns[0]])] != "OTHER"
                else "X"
            )
            > 0.0
            for r in feats
        ]
    )


def load_bond_head(ckpt: str, device: torch.device) -> nn.Module:
    """Load a trained head. Accepts a Lightning checkpoint or a bare state dict."""
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
    state = {k.removeprefix("model."): v for k, v in state.items()}
    head = BondHead()
    head.load_state_dict(state)
    return head.eval().to(device)


def bond_jaccard(
    predicted: Sequence[tuple[int, int]], true: Sequence[tuple[int, int]]
) -> float:
    """Agreement between two bond graphs, as sets of unordered pairs."""
    a = {tuple(sorted(map(int, p))) for p in predicted}
    b = {tuple(sorted(map(int, p))) for p in true}
    return len(a & b) / max(len(a | b), 1)

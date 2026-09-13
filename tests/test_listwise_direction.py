"""The listwise term must sharpen the good end of whichever label it is given.

The ListNet target is a softmax over the labels, so its mass lands on one end
of each group and that is where the gradient goes. RMSD is good when small, pK
is good when large, and the same expression cannot serve both. Getting the sign
wrong does not break training -- the ordering constraint is symmetric -- it
quietly spends the term on telling the WEAKEST binders apart rather than the
tightest -- the ordering it rewards is unchanged, the end it pays attention to
is not.
"""

from __future__ import annotations

import torch

from prolit.config import ProLITMLMConfig, RescoreTrainingConfig
from prolit.model.rescore_module import ComplexRescoreModule


def _module(*, higher_is_better: bool) -> ComplexRescoreModule:
    config = RescoreTrainingConfig(model=ProLITMLMConfig(atom_codebook_size=64))
    config.listwise_higher_is_better = higher_is_better
    return ComplexRescoreModule(config)


def _loss(module: ComplexRescoreModule, pred: list[float], label: list[float]) -> float:
    groups = torch.zeros(len(pred), dtype=torch.long)
    return float(
        module._listwise_loss(  # noqa: SLF001
            torch.tensor(pred), torch.tensor(label), groups
        )
    )


def test_default_rewards_predicting_the_smallest_label_best() -> None:
    """RMSD semantics: the pose with the lowest label should score lowest."""
    module = _module(higher_is_better=False)
    labels = [0.2, 2.0, 5.0]
    aligned = _loss(module, [0.2, 2.0, 5.0], labels)
    reversed_ = _loss(module, [5.0, 2.0, 0.2], labels)
    assert aligned < reversed_


def test_flag_rewards_predicting_the_largest_label_best() -> None:
    """pK semantics: the tightest binder should score highest."""
    module = _module(higher_is_better=True)
    labels = [4.0, 6.0, 9.0]
    aligned = _loss(module, [4.0, 6.0, 9.0], labels)
    reversed_ = _loss(module, [9.0, 6.0, 4.0], labels)
    assert aligned < reversed_


def test_the_sign_decides_which_end_of_the_group_is_worth_getting_right() -> None:
    """This is what the flag actually changes, and it is not the ordering.

    Negating the labels and the predictions together leaves the correspondence
    between them intact, so BOTH signs reward a correctly ordered group -- the
    wrong sign does not train a head to rank binders backwards. What it changes
    is where the softmax puts its mass, and therefore which mistakes the term
    is willing to pay for.

    Both candidates below get one pair of a three-ligand group wrong. ``top``
    has the tightest binder in the right place and the two weak ones swapped;
    ``bottom`` has the weakest in the right place and the two tight ones
    swapped. Ranking power cares about the whole ordering, but capacity is
    finite, and it is the tight end that separates a useful scoring function
    from a useless one.
    """
    labels = [4.0, 6.0, 9.0]  # pK: 9.0 is the tightest binder
    top = [6.0, 4.0, 9.0]  # tightest placed right, weak pair swapped
    bottom = [4.0, 9.0, 6.0]  # weakest placed right, tight pair swapped

    on = _module(higher_is_better=True)
    assert _loss(on, top, labels) < _loss(on, bottom, labels)

    off = _module(higher_is_better=False)
    assert _loss(off, bottom, labels) < _loss(off, top, labels)


def test_the_default_is_the_published_behaviour() -> None:
    """Existing pose heads were trained without this flag; it must default off."""
    assert RescoreTrainingConfig().listwise_higher_is_better is False

"""Every arm named anywhere in the registry must be resolvable.

``SEED_FAMILIES`` names arms by string. ``report.seed_families`` only uses those
strings to build directory paths, so a name that is not in ``REGISTRY`` costs
nothing locally -- the tables still render. The failure lands in
``scripts/infer_affinity.py``, which calls ``variants.get()``, and therefore on
the cluster: after the job has queued, after the head has trained, in the
evaluation that was supposed to produce the number. That happened once while
adding the clean-union arms.
"""

from __future__ import annotations

import pytest

from affinity_bench import variants


def test_every_seed_family_arm_is_registered() -> None:
    missing = [
        arm
        for family in variants.SEED_FAMILIES.values()
        for arm in family.values()
        if arm not in variants.REGISTRY
    ]
    assert not missing, f"named in SEED_FAMILIES but absent from REGISTRY: {missing}"


def test_the_paired_reference_is_itself_a_family() -> None:
    assert variants.SEED_REFERENCE in variants.SEED_FAMILIES


def test_seed_family_arms_declare_a_head_to_evaluate() -> None:
    """An arm in a family is meant to be run; one with no head cannot be."""
    headless = [
        arm
        for family in variants.SEED_FAMILIES.values()
        for arm in family.values()
        if not variants.get(arm).heads
    ]
    assert not headless, f"in SEED_FAMILIES with no head checkpoint: {headless}"


def test_arm_order_names_only_registered_arms() -> None:
    unknown = [a for a in variants.ARM_ORDER if a not in variants.REGISTRY]
    assert not unknown, f"ARM_ORDER names unregistered arms: {unknown}"


def test_registry_keys_match_their_variant_names() -> None:
    mismatched = [k for k, v in variants.REGISTRY.items() if k != v.name]
    assert not mismatched, f"registry key != Variant.name for: {mismatched}"


def test_get_rejects_an_unknown_arm_by_name() -> None:
    with pytest.raises(KeyError, match="unknown affinity arm"):
        variants.get("no_such_arm")


def test_codebook_size_follows_the_shared_arm_registry() -> None:
    """An arm's code space must be the one its tokenizer actually has.

    ``prolit_bench.variants`` is the single definition of what a tokenizer arm
    means -- which runs, which normalization statistics, how large the code
    space is -- and every benchmark is supposed to follow it rather than restate
    it. ``benchmarks/test_variant_agreement.py`` asserted this for affinity
    until affinity moved out of that bench; the check moved here with it rather
    than being dropped.

    Getting it wrong does not raise: a head built against a vocabulary the
    weights were never fitted to still produces 285 plausible numbers.
    """
    from prolit_bench import variants as shared  # noqa: PLC0415

    for name, arm in variants.REGISTRY.items():
        expected = (
            shared.SEPARATE if arm.is_separate else shared.JOINT
        ).combined_codebook_size
        assert arm.codebook_size == expected, (
            f"{name}: declares codebook_size {arm.codebook_size}, but its "
            f"tokenizer's combined code space is {expected}"
        )

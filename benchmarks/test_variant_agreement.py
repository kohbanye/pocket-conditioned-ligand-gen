"""The benchmarks must agree on what a tokenizer arm is.

Three benchmarks report on the same arms in three paper tables. They used to
resolve those arms independently, and had silently drifted apart. These tests
pin the parts that must match and make the one part that still differs -- which
checkpoint of a run each table published -- visible instead of hidden.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# recon_bench is outside the workspace, so it is not installed into this
# environment; add it to the path so its arm table can be compared here.
BENCHMARKS = Path(__file__).resolve().parent
if str(BENCHMARKS / "recon-bench") not in sys.path:
    sys.path.insert(0, str(BENCHMARKS / "recon-bench"))

from prolit_bench import variants  # noqa: E402

ARM_NAMES = ["joint", "separate", "separate_4096"]

#: Trained weights are not in git. The identity checks below are pure metadata
#: and always run; only the two that resolve an arm to a file need them present.
needs_weights = pytest.mark.skipif(
    not variants.VQ_RUNS_DIR.exists(),
    reason="no local training runs (weights are not tracked in git)",
)


def _recon_bench_arms() -> dict:
    """recon_bench's own arm table, keyed by the shared registry's names."""
    recon_bench_arms = pytest.importorskip(
        "recon_bench.adapters.own_allatom", reason="recon_bench not importable here"
    ).ARMS
    return {
        variants.ALIASES.get(name, name): arm
        for name, arm in recon_bench_arms.items()
        # recon_bench also carries binning / ligand-own-frame arms, which are
        # reconstruction-only and have no rescoring or generation counterpart.
        if variants.ALIASES.get(name, name) in variants.REGISTRY
    }


def _tail(path: Path) -> tuple[str, ...]:
    """The part of a cache path after the repository root it was resolved from.

    Comparing absolute paths would only prove the two packages were loaded from
    the same checkout; comparing the tail proves they name the same file, which
    is the thing that has to agree.
    """
    parts = Path(path).resolve().parts
    return parts[parts.index("data") :] if "data" in parts else parts


@pytest.mark.parametrize("name", ARM_NAMES)
def test_recon_bench_arm_identity_matches_registry(name: str) -> None:
    """Run names and normalization statistics must be identical."""
    theirs = _recon_bench_arms()[name]
    ours = variants.get(name)
    assert theirs.protein_run == ours.protein_run
    assert theirs.ligand_run == ours.ligand_run
    assert _tail(theirs.protein_norm) == _tail(ours.protein_norm)
    assert _tail(theirs.ligand_norm) == _tail(ours.ligand_norm)


@pytest.mark.parametrize("name", ARM_NAMES)
def test_pose_rescoring_bench_arm_codebook_size_matches_registry(name: str) -> None:
    """The bench states the COMBINED code space; it must follow from the arm."""
    rescoring_variants = pytest.importorskip(
        "pose_rescoring_bench.variants",
        reason="pose-rescoring-bench not importable here",
    )
    variant = rescoring_variants.get(name)
    ours = variants.get(name)
    for task in (variant.rescoring, variant.affinity):
        if task is None:
            continue
        assert task.codebook_size == ours.combined_codebook_size, (
            f"{name}: pose_rescoring_bench {task.codebook_size} vs registry "
            f"{ours.combined_codebook_size}"
        )


@needs_weights
@pytest.mark.parametrize("name", ARM_NAMES)
def test_selection_policies_are_recorded_not_guessed(name: str) -> None:
    """Every task's published policy resolves to a file that exists."""
    for policy in set(variants.PUBLISHED_POLICY.values()):
        resolved = variants.checkpoints(name, policy)
        for key, path in resolved.items():
            assert path.exists(), f"{name}/{policy}: missing {key} -> {path}"


@needs_weights
def test_known_policy_divergence_is_reported() -> None:
    """Document, as an assertion, where the two policies actually differ.

    This is not a failure: it records the state the paper's tables were computed
    in. If a future change makes the policies agree, this test fails and should
    be deleted along with ``PUBLISHED_POLICY``.
    """
    diverging = [
        f"{name}.{key}"
        for name in ARM_NAMES
        for key in ("protein_ckpt", "ligand_ckpt")
        if variants.checkpoints(name, "best")[key].resolve()
        != variants.checkpoints(name, "last")[key].resolve()
    ]

    assert diverging, (
        "'best' and 'last' now resolve identically for every arm -- the "
        "reconstruction and rescoring tables no longer disagree. Drop "
        "PUBLISHED_POLICY and this test."
    )
    # The joint arm agreed all along; only the separate arms diverge.
    assert not any(d.startswith("joint.") for d in diverging), (
        f"the joint arm should not diverge, but got {diverging}"
    )

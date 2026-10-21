"""The adapter contract, checked without any baseline environment present.

These tests never score anything: the baselines need interpreters and GPUs that
CI does not have. What they do check is the part that silently breaks a table --
the dump schema, and the refusal to report a comparison over a reduced target
set.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pose_rescoring_bench import adapters
from pose_rescoring_bench.adapters.base import DUMP_COLUMNS, PoseBaseline
from pose_rescoring_bench.io_dumps import POSE_COLUMNS


def test_every_registered_name_builds() -> None:
    """Construction must not need the baseline's environment."""
    for name in adapters.available():
        built = adapters.build(name)
        assert isinstance(built, PoseBaseline)
        assert built.name == name


def test_unknown_name_names_the_alternatives() -> None:
    with pytest.raises(KeyError, match="rtmscore"):
        adapters.build("nosuchmodel")


def test_dump_schema_is_a_subset_of_the_canonical_pose_dump() -> None:
    """Adapters must write columns the metric layer already knows how to read."""
    assert set(DUMP_COLUMNS) <= set(POSE_COLUMNS)
    assert set(DUMP_COLUMNS) == {"pdbid", "pose", "rmsd", "head"}


def test_contact_flags_are_declared() -> None:
    """The flag that decides which rows are like-for-like is never left default.

    Every baseline in this table is handed explicit interaction geometry; our
    arms are not. A future adapter that is not must say so deliberately.
    """
    for name in adapters.available():
        assert adapters.build(name).uses_explicit_contacts is True


def test_coverage_failure_names_the_missing_targets() -> None:
    frame = pd.DataFrame(
        {"pdbid": ["1a30", "1a30"], "pose": ["a", "b"],
         "rmsd": [0.5, 3.0], "head": [1.0, 0.0]}
    )
    with pytest.raises(ValueError, match="1bcu"):
        PoseBaseline.check_coverage(frame, ["1a30", "1bcu"])


def test_coverage_passes_when_every_target_is_present() -> None:
    frame = pd.DataFrame(
        {"pdbid": ["1a30", "1bcu"], "pose": ["a", "b"],
         "rmsd": [0.5, 3.0], "head": [1.0, 0.0]}
    )
    PoseBaseline.check_coverage(frame, ["1a30", "1bcu"])


def test_only_vina_is_empirical() -> None:
    """The reference row is marked as one, so a table can separate it out."""
    empirical = {n for n in adapters.available() if adapters.build(n).is_empirical}
    assert empirical == {"vina"}

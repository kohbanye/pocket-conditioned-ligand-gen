"""Every config attribute the rescoring datamodule reads must exist by default.

``RescoreDataModule.setup`` reads its options off the config unconditionally,
while ``pipelines/train/scoring_head.py`` sets some of them only when the
matching CLI flag is passed. When those two drift, the run does not warn and
does not degrade -- it dies inside Lightning's setup hook with an
``AttributeError`` that names a field, and it dies only for the runs that did
*not* pass the flag. ``drop_native_pose`` drifted exactly that way: 128 of the
130 recorded ``scoring_head.py`` commands in ``jobs/`` could not start.

A dataclass default is also what lets a config pickled into an older checkpoint
resolve a newly added attribute, so this is the shape that keeps existing heads
loadable.
"""

from __future__ import annotations

import ast
from pathlib import Path

from prolit.config import RescoreTrainingConfig

_DATASET = Path(__file__).resolve().parents[1] / "src/prolit/data/rescore_dataset.py"


def _config_attributes_read(source: Path) -> set[str]:
    """Names read as ``self.config.<name>`` anywhere in a module."""
    tree = ast.parse(source.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        inner = node.value
        if (
            isinstance(inner, ast.Attribute)
            and inner.attr == "config"
            and isinstance(inner.value, ast.Name)
            and inner.value.id == "self"
        ):
            found.add(node.attr)
    return found


def test_datamodule_reads_nothing_the_config_lacks() -> None:
    cfg = RescoreTrainingConfig()
    missing = sorted(
        name for name in _config_attributes_read(_DATASET) if not hasattr(cfg, name)
    )
    assert not missing, (
        f"rescore_dataset.py reads {missing} off its config, but "
        f"RescoreTrainingConfig does not define them. Any scoring_head.py run "
        f"that does not set them by hand dies in Lightning's setup hook."
    )


def test_drop_native_pose_defaults_to_keeping_it() -> None:
    """The default has to match what the published heads were trained with.

    Only 2 of the 130 recorded commands pass ``--drop-native-pose``, so the
    default is the setting almost every head used, and changing it would
    silently retrain them on a different corpus.
    """
    assert RescoreTrainingConfig().drop_native_pose is False

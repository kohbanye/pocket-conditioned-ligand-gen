"""The stapled arm must not be able to change how the ProLIT arm is called.

``_score_target`` hands the target id to encoders that ask for it and to no
others, gated on ``needs_struct_id``. If ``PoseEncoder`` ever grew that
attribute, or the stapled façade lost it, the dispatch would silently flip: the
ProLIT path would start receiving an extra positional argument (a loud failure)
or the stapled path would stop receiving the id it needs to find its pocket
codes and would report a cache miss for every target (a quiet one, and the kind
that reads as "the baseline could not be scored").

These numbers are published, so the invariant is pinned rather than assumed.
"""

from __future__ import annotations

import inspect

from prolit.tokenizers.pose_encoder import PoseEncoder

from pose_rescoring_bench.inference.stapled import StapledPoseEncoder


def test_prolit_encoder_does_not_ask_for_a_struct_id() -> None:
    assert getattr(PoseEncoder, "needs_struct_id", False) is False
    params = list(inspect.signature(PoseEncoder.setup_pocket).parameters)
    assert params == ["self", "protein_text", "reference_heavy"]


def test_stapled_encoder_asks_for_a_struct_id() -> None:
    assert StapledPoseEncoder.needs_struct_id is True
    params = list(inspect.signature(StapledPoseEncoder.setup_pocket).parameters)
    assert params == ["self", "protein_text", "reference_heavy", "struct_id"]


def test_both_encoders_expose_the_same_per_pose_call() -> None:
    """``_score_pose`` calls ``ligand_seq(pocket, mol, frame)`` for either arm."""
    for cls in (PoseEncoder, StapledPoseEncoder):
        params = list(inspect.signature(cls.ligand_seq).parameters)
        assert len(params) == 4, f"{cls.__name__}.ligand_seq takes {params}"
        assert params[0] == "self"

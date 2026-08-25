"""A run must record what produced it, beside what it produced.

Weights are not in git and neither are the job scripts that launched them, so
``run.json`` in the run directory is the only thing linking a checkpoint back to
a command and a commit. If it stops being written, that link is gone silently --
the checkpoints still look fine.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest  # noqa: TC002

from prolit.provenance import (
    MANIFEST_NAME,
    RecordProvenance,
    git_state,
    read_manifest,
    run_manifest,
    write_manifest,
)

REPO = Path(__file__).resolve().parent.parent


def test_manifest_records_the_command_and_seed() -> None:
    manifest = run_manifest(seed=11)
    assert manifest["command"][0] == sys.argv[0]
    assert manifest["seed"] == 11
    assert manifest["hostname"]
    assert manifest["started"].endswith("+00:00")


def test_git_state_reports_sha_and_dirtiness() -> None:
    state = git_state(REPO)
    assert state["sha"] is not None
    assert len(state["sha"]) == 40
    assert isinstance(state["dirty"], bool)


def test_git_state_outside_a_checkout_degrades(tmp_path: Path) -> None:
    """A run somewhere without git should still record everything else."""
    state = git_state(tmp_path)
    assert state["sha"] is None


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    written = write_manifest(tmp_path / "run", seed=3)
    assert written.name == MANIFEST_NAME
    assert read_manifest(tmp_path / "run")["seed"] == 3
    assert json.loads(written.read_text())["seed"] == 3


def test_rerunning_overwrites_rather_than_accumulates(tmp_path: Path) -> None:
    """A resumed run describes how it was last run; the checkpoint did too."""
    write_manifest(tmp_path, seed=1)
    write_manifest(tmp_path, seed=2)
    assert read_manifest(tmp_path)["seed"] == 2
    assert len(list(tmp_path.glob("run*.json"))) == 1


def test_read_manifest_missing_is_none(tmp_path: Path) -> None:
    """Runs that predate this feature must not raise when inspected."""
    assert read_manifest(tmp_path) is None


def test_job_metadata_is_picked_up_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """jobs/submit.py cannot know the run dir, so it passes this through env."""
    monkeypatch.setenv("PROLIT_JOB_NAME", "lm_pre")
    monkeypatch.setenv("PROLIT_JOB_RESOURCE", "node_f")
    monkeypatch.setenv("PROLIT_JOB_HOURS", "8")
    job = run_manifest()["job"]
    assert job["name"] == "lm_pre"
    assert job["resource"] == "node_f"


def test_sweep_point_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Which arm of a comparison this run is -- what _v8 / _v10 names encoded."""
    monkeypatch.setenv("PROLIT_JOB_NAME", "aff_pooling-attn")
    monkeypatch.setenv("PROLIT_JOB_SWEEP", '{"pooling": "attn"}')
    assert run_manifest()["job"]["sweep"] == {"pooling": "attn"}


def test_an_unreadable_sweep_does_not_lose_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run still ran; recording it verbatim beats recording nothing."""
    monkeypatch.setenv("PROLIT_JOB_NAME", "aff")
    monkeypatch.setenv("PROLIT_JOB_SWEEP", "not json")
    job = run_manifest()["job"]
    assert job["sweep"] == "not json"
    assert job["name"] == "aff"


def test_no_job_metadata_for_an_interactive_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in (
        "PROLIT_JOB_NAME",
        "PROLIT_JOB_RESOURCE",
        "PROLIT_JOB_HOURS",
        "PROLIT_JOB_SWEEP",
    ):
        monkeypatch.delenv(var, raising=False)
    assert "job" not in run_manifest()


def test_callback_writes_into_the_checkpoint_dir(tmp_path: Path) -> None:
    """The manifest belongs where the weights land, not where the job ran."""

    class _Ckpt:
        dirpath = str(tmp_path / "checkpoints")

    class _Trainer:
        checkpoint_callback = _Ckpt()
        global_rank = 0

    cb = RecordProvenance(seed=5)
    cb.setup(_Trainer(), None, "fit")
    assert read_manifest(_Ckpt.dirpath)["seed"] == 5


def test_callback_is_silent_off_rank_zero(tmp_path: Path) -> None:
    """Under DDP only one process should write it."""

    class _Ckpt:
        dirpath = str(tmp_path / "checkpoints")

    class _Trainer:
        checkpoint_callback = _Ckpt()
        global_rank = 1

    RecordProvenance(seed=5).setup(_Trainer(), None, "fit")
    assert not Path(_Ckpt.dirpath).exists()


def test_every_training_script_records_provenance() -> None:
    """The part that rots: a new trainer added without the callback.

    ``write_manifest(`` counts too. The callback reads its output directory off
    a ``ModelCheckpoint``, so a script that deliberately writes no checkpoints
    -- a hyperparameter search whose trials are thrown away -- cannot use it and
    calls the writer directly instead. What the rule is actually about is that
    the run records the command and SHA that produced it, not which of the two
    ways it does so.
    """
    offenders = [
        p.name
        for p in sorted((REPO / "pipelines" / "train").glob("*.py"))
        if not any(
            call in p.read_text() for call in ("RecordProvenance(", "write_manifest(")
        )
    ]
    assert not offenders, f"train without recording provenance: {offenders}"


def test_submit_passes_job_metadata_to_the_run() -> None:
    """Without this the manifest cannot say which submission produced it."""
    text = (REPO / "jobs" / "submit.py").read_text()
    for var in (
        "PROLIT_JOB_NAME",
        "PROLIT_JOB_RESOURCE",
        "PROLIT_JOB_HOURS",
        "PROLIT_JOB_SWEEP",
    ):
        assert var in text, var

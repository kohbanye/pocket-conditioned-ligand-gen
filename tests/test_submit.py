"""Sweeps must expand to exactly the jobs the command describes.

``--sweep`` is the reason job scripts stop multiplying: 22 of the archived ones
differed only in the value of one flag. That only holds if the expansion is
trustworthy, because the failure mode is quiet -- a job that runs to completion
with the wrong hyper-parameter looks exactly like a job that ran correctly, and
the number it produces goes into a table.

``jobs/`` is not a package (it is site-specific and mostly git-ignored), so the
module is loaded by path rather than imported.
"""

from __future__ import annotations

import importlib.util
import shlex
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_submit():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location(
        "_jobs_submit", REPO / "jobs" / "submit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


submit = _load_submit()


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *argv: str) -> list[Path]:
    """Run the CLI with a throwaway output directory; return what it wrote."""
    monkeypatch.setattr(submit, "GENERATED", tmp_path)
    monkeypatch.setattr(sys, "argv", ["submit.py", *argv])
    submit.main()
    return sorted(tmp_path.glob("*.sh"))


def _command_of(script: Path) -> list[str]:
    """The argv a generated script runs, from its ``"$PY" ...`` line."""
    line = next(
        ln for ln in script.read_text().splitlines() if ln.startswith('"$PY" ')
    )
    return shlex.split(line)[1:]


def test_parse_sweep_reads_one_axis() -> None:
    assert submit.parse_sweep(["pooling=mean,attn"]) == {"pooling": ["mean", "attn"]}


def test_parse_sweep_rejects_a_spec_without_values() -> None:
    for spec in ("pooling", "pooling=", "=mean"):
        with pytest.raises(ValueError, match="--sweep"):
            submit.parse_sweep([spec])


def test_parse_sweep_rejects_a_repeated_value() -> None:
    """Two identical points would collide on one filename and one job name."""
    with pytest.raises(ValueError, match="repeats"):
        submit.parse_sweep(["pooling=mean,mean"])


def test_parse_sweep_rejects_the_same_axis_twice() -> None:
    with pytest.raises(ValueError, match="twice"):
        submit.parse_sweep(["lr=1e-4", "lr=1e-3"])


def test_expand_is_the_cross_product() -> None:
    points = submit.expand({"a": ["1", "2"], "b": ["x", "y", "z"]})
    assert len(points) == 6
    assert points[0] == {"a": "1", "b": "x"}
    assert points[-1] == {"a": "2", "b": "z"}
    # First axis slowest, so the readable names group by it.
    assert [p["a"] for p in points] == ["1", "1", "1", "2", "2", "2"]


def test_expand_of_nothing_is_one_plain_job() -> None:
    assert submit.expand({}) == [{}]


def test_substitute_replaces_the_placeholder() -> None:
    cmd = ["train.py", "--pooling", "{pooling}", "--out", "runs/{pooling}"]
    assert submit.substitute(cmd, {"pooling": "attn"}) == [
        "train.py", "--pooling", "attn", "--out", "runs/attn",
    ]


def test_substitute_rejects_an_undefined_placeholder() -> None:
    """A typo must fail here, not silently reach the training script."""
    with pytest.raises(ValueError, match="poooling"):
        submit.substitute(["train.py", "{poooling}"], {"pooling": "attn"})


def test_substitute_leaves_braces_alone_without_a_sweep() -> None:
    """A plain job is unaffected by the placeholder syntax existing."""
    cmd = ["train.py", "--filter", "{unrelated}"]
    assert submit.substitute(cmd, {}) == cmd


def test_names_carry_the_swept_values() -> None:
    names = submit.name_points("aff", submit.expand({"pooling": ["mean", "attn"]}))
    assert names == ["aff_pooling-mean", "aff_pooling-attn"]


def test_names_sanitise_values_for_the_filesystem() -> None:
    names = submit.name_points("tok", submit.expand({"dir": ["data/a b", "c"]}))
    assert names == ["tok_dir-data-a-b", "tok_dir-c"]


def test_names_fall_back_to_numbering_when_too_long() -> None:
    """A name is only useful while it can be read."""
    long = ["data/lm_tokens_pretrain_mixed_allatom_v" + str(i) for i in range(2)]
    names = submit.name_points("tok", submit.expand({"token_dir": long}))
    assert names == ["tok_1", "tok_2"]


def test_a_single_point_sweep_still_carries_its_value() -> None:
    """One value is a degenerate sweep, but still an arm worth naming."""
    assert submit.name_points("aff", [{"pooling": "attn"}]) == ["aff_pooling-attn"]


def test_a_plain_job_still_writes_one_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written = _run(
        tmp_path, monkeypatch,
        "--name", "lm_pre", "--resource", "gpu_1", "--hours", "8",
        "--", "pipelines/train/clm.py", "--token-dir", "data/x",
    )
    assert [p.name for p in written] == ["lm_pre.sh"]
    text = written[0].read_text()
    assert "pipelines/train/clm.py --token-dir data/x" in text
    assert "PROLIT_JOB_SWEEP" not in text


def test_a_sweep_writes_one_script_per_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written = _run(
        tmp_path, monkeypatch,
        "--name", "aff", "--resource", "gpu_1", "--hours", "8",
        "--sweep", "tau=0.3,0.5,0.7",
        "--", "pipelines/train/scoring_head.py", "--listwise-tau", "{tau}",
    )
    assert [p.name for p in written] == [
        "aff_tau-0.3.sh", "aff_tau-0.5.sh", "aff_tau-0.7.sh",
    ]
    for path in written:
        tau = path.stem.split("-")[-1]
        text = path.read_text()
        assert f"--listwise-tau {tau}" in text
        assert "{tau}" not in text
        assert f"export PROLIT_JOB_NAME={path.stem}" in text
        assert f'PROLIT_JOB_SWEEP=\'{{"tau": "{tau}"}}\'' in text
        assert f"#$ -N {path.stem}" in text


def test_a_cross_product_covers_every_combination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written = _run(
        tmp_path, monkeypatch,
        "--name", "g", "--resource", "cpu_4", "--hours", "1",
        "--sweep", "a=1,2", "--sweep", "b=x,y",
        "--", "train.py", "--a", "{a}", "--b", "{b}",
    )
    assert len(written) == 4
    ran = {tuple(_command_of(p)[1:]) for p in written}
    assert ran == {
        ("--a", "1", "--b", "x"), ("--a", "1", "--b", "y"),
        ("--a", "2", "--b", "x"), ("--a", "2", "--b", "y"),
    }


def test_a_mistyped_placeholder_names_the_typo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Both checks fire on a typo; the one that names the mistake must win."""
    with pytest.raises(SystemExit):
        _run(
            tmp_path, monkeypatch,
            "--name", "g", "--resource", "cpu_4", "--hours", "1",
            "--sweep", "pooling=mean,attn",
            "--", "train.py", "--pooling", "{poooling}",
        )
    assert "{poooling}" in capsys.readouterr().err


def test_a_sweep_nothing_uses_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise it silently submits N copies of one job."""
    with pytest.raises(SystemExit):
        _run(
            tmp_path, monkeypatch,
            "--name", "g", "--resource", "cpu_4", "--hours", "1",
            "--sweep", "pooling=mean,attn",
            "--", "train.py", "--pooling", "mean",
        )
    assert not list(tmp_path.glob("*.sh"))


def test_an_oversized_sweep_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cross product is one flag to write and N nodes to pay for."""
    with pytest.raises(SystemExit):
        _run(
            tmp_path, monkeypatch,
            "--name", "g", "--resource", "node_f", "--hours", "24",
            "--sweep", "a=1,2,3", "--max-jobs", "2",
            "--", "train.py", "--a", "{a}",
        )
    assert not list(tmp_path.glob("*.sh"))


def test_the_billing_estimate_counts_every_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The multiplier is where a sweep gets expensive, so it must be stated."""
    _run(
        tmp_path, monkeypatch,
        "--name", "g", "--resource", "gpu_1", "--hours", "8",
        "--sweep", "a=1,2,3",
        "--", "train.py", "--a", "{a}",
    )
    out = capsys.readouterr().out
    assert "6.00 node-hours" in out  # 0.25 * 8 * 3, not 2.00
    assert "not submitted" in out


def test_generated_scripts_are_not_submitted_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """qsub happens only on request; the rule is agree first, submit second."""

    def _boom(*_args: object, **_kwargs: object) -> None:
        msg = "submit.py must not qsub without --submit"
        raise AssertionError(msg)

    monkeypatch.setattr(submit.subprocess, "run", _boom)
    _run(
        tmp_path, monkeypatch,
        "--name", "g", "--resource", "cpu_4", "--hours", "1",
        "--sweep", "a=1,2", "--", "train.py", "--a", "{a}",
    )

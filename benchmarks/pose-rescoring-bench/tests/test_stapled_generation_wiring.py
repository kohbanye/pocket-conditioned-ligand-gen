"""The stapled generation arm must not be able to change how ProLIT generates.

Generation runs through the sbdd-bench ``own`` adapter, which is selected by
environment variables. A leak in either direction is silent and expensive: a
ProLIT run that picked up ``SBDD_OWN_MODE=stapled`` would call the wrong
generator, and a stapled run that fell through to ``allatom`` would look for a
VQ-VAE this arm does not have and fail 100 targets on it.

``e250_gen`` is the published ProLIT generation arm, so its env is pinned here
against exactly the fields the stapled branch adds.
"""

from __future__ import annotations

from pose_rescoring_bench.config import EvalConfig
from pose_rescoring_bench.inference.generation import _own_env
from pose_rescoring_bench.variants import get

_STAPLED_KEYS = (
    "SBDD_OWN_ESM3_CACHE",
    "SBDD_OWN_STAPLED_VOCAB",
    "SBDD_OWN_CONFSEQ_REPO",
)


def _env(variant: str) -> dict[str, str]:
    cfg = EvalConfig()
    return _own_env(get(variant).generation, cfg.paths, cfg.generation)


def test_stapled_selects_its_own_generator() -> None:
    env = _env("stapled")
    assert env["SBDD_OWN_MODE"] == "stapled"
    for key in _STAPLED_KEYS:
        assert env[key], f"{key} is empty"
    # No VQ-VAE and no normalization statistics: there is no such thing in this
    # arm, and passing a stale one would silently decode against the wrong
    # codebook rather than fail.
    assert "SBDD_OWN_VQVAE_CKPT" not in env
    assert "SBDD_OWN_NORM_STATS" not in env


def test_prolit_generation_is_untouched() -> None:
    env = _env("e250_gen")
    assert env["SBDD_OWN_MODE"] == "allatom"
    assert env["SBDD_OWN_VQVAE_CKPT"]
    for key in _STAPLED_KEYS:
        assert key not in env, f"{key} leaked into the ProLIT arm"


def test_only_the_stapled_variant_declares_itself_stapled() -> None:
    assert get("stapled").generation.is_stapled is True
    assert get("e250_gen").generation.is_stapled is False

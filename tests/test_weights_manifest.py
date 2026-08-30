"""The weight manifest must stay consistent with what the benchmark reads.

Nothing here downloads: the point is that a typo in a group name or an env var
is caught locally rather than on the machine that has no weights yet.
"""

from __future__ import annotations

from pathlib import Path

from prolit.weights import ENV_FOR, FILES, GROUPS, env_lines


def test_groups_only_name_known_files() -> None:
    for group, names in GROUPS.items():
        unknown = set(names) - set(FILES)
        assert not unknown, f"group {group!r} names unknown weights: {unknown}"


def test_env_map_only_names_known_files() -> None:
    assert not set(ENV_FOR) - set(FILES)


def test_tokenizer_is_a_set() -> None:
    """Every group that has the VQ-VAE must also carry its normalization stats.

    Pairing a checkpoint with the wrong stats does not raise -- it produces
    plausible coordinates at the wrong scale. The grouping is what stops a
    caller assembling the pair by hand.
    """
    for group, names in GROUPS.items():
        if "atom_vqvae" in names:
            assert "norm_stats" in names, f"group {group!r} ships a VQ-VAE alone"


def test_generate_group_covers_the_adapter() -> None:
    """The default group must set every variable the own adapter needs to run."""
    needed = {"SBDD_OWN_VQVAE_CKPT", "SBDD_OWN_NORM_STATS", "SBDD_OWN_LM_CKPT"}
    paths = dict.fromkeys(GROUPS["generate"], Path("x"))
    got = {
        line.split("=")[0].removeprefix("export ") for line in env_lines(paths)
    }
    assert needed <= got, f"missing {needed - got}"

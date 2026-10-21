"""Every registered baseline adapter must be buildable and honest about coverage.

The failure this guards against is the expensive kind: an adapter that imports
fine here but only breaks inside its own subprocess, on a GPU node, after the
run has started -- or worse, one that returns a dump missing the complexes its
model could not handle, which then reads as a full-coverage comparison.
"""

from __future__ import annotations

import pandas as pd
import pytest

from affinity_bench import adapters
from affinity_bench.io_dumps import AFFINITY_COLUMNS


def test_every_registered_adapter_builds() -> None:
    """Lazy imports mean a typo in the registry is invisible until run time."""
    for name in adapters.available():
        model = adapters.build(name)
        assert model.name == name, f"{name}: adapter reports name {model.name!r}"


def test_build_rejects_an_unknown_baseline() -> None:
    with pytest.raises(KeyError, match="unknown affinity baseline"):
        adapters.build("no_such_baseline")


def test_dump_columns_match_the_dump_schema() -> None:
    """An adapter returning a different column order silently mislabels a dump."""
    # The dump schema carries `pll` as a zero-shot reference the tables do not
    # use; an adapter supplies everything else.
    expected = tuple(c for c in AFFINITY_COLUMNS if c != "pll")
    assert tuple(adapters.DUMP_COLUMNS) == expected, adapters.DUMP_COLUMNS


def test_every_adapter_declares_whether_it_uses_explicit_contacts() -> None:
    """The flag decides which rows are a like-for-like peer of our tokenizer.

    Our model is handed no hand-specified contacts, so a method that featurises
    interactions directly is measuring something else. Leaving this unset would
    put those rows in the table with no way to tell them apart.
    """
    for name in adapters.available():
        model = adapters.build(name)
        assert isinstance(model.uses_explicit_contacts, bool), name


def test_check_coverage_rejects_a_short_dump() -> None:
    from pathlib import Path  # noqa: PLC0415

    from affinity_bench.inference import Complex  # noqa: PLC0415

    wanted = [
        Complex(pdbid, Path("p.pdb"), Path("l.sdf"), 5.0, "1")
        for pdbid in ("1a30", "1bcu", "1bzc")
    ]
    full = pd.DataFrame(
        {"pdbid": [c.pdbid for c in wanted], "logka": 5.0, "cluster": "1", "head": 5.0}
    )
    adapters.AffinityBaseline.check_coverage(full, wanted)

    short = full.iloc[:2]
    with pytest.raises(ValueError, match="absent from the dump"):
        adapters.AffinityBaseline.check_coverage(short, wanted)

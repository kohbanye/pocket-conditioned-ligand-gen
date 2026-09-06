"""One pathological ligand must cost one ligand, not the shard it sits in.

A decoy shard walks tens of thousands of complexes in a single process. RDKit's
``GetSubstructMatches`` over a highly symmetric ligand explores a factorial
space and ``maxMatches`` caps the results it keeps, not the search it does, so
one molecule can wedge the whole run: a shard did exactly that, spinning a core
for eleven hours after its last output at 04:55 and hitting its walltime with
nothing further written.

What this cannot do is worth pinning as clearly as what it can. A Python signal
handler runs between bytecodes, so a call that stays inside C indefinitely is
not interrupted. The other half of the fix -- checkpointing meta.json after
every bucket -- is what makes a wedged shard's finished work readable anyway,
and is tested through the builder rather than here.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

# pipelines modules import each other by bare name, so its directory goes on the
# path the same way the builders themselves do.
_CORPORA = Path(__file__).resolve().parents[1] / "pipelines" / "corpora"
if str(_CORPORA) not in sys.path:
    sys.path.insert(0, str(_CORPORA))


@pytest.fixture(scope="module")
def budget_cls() -> type:
    from tokenize_decoys import _SiteBudget  # noqa: PLC0415

    return _SiteBudget


def test_work_over_budget_is_abandoned(budget_cls: type) -> None:
    """And as a TimeoutError, which the builder's `except Exception` skips on."""
    b = budget_cls(1)
    b.arm()
    try:
        with pytest.raises(TimeoutError):
            time.sleep(3)
    finally:
        b.disarm()
    assert b.hit == 1


def test_work_inside_the_budget_runs_untouched(budget_cls: type) -> None:
    b = budget_cls(1)
    b.arm()
    time.sleep(0.05)
    b.disarm()
    assert b.hit == 0
    # And the timer really is cancelled: a later sleep past the budget must not
    # inherit an armed alarm, or one slow complex would poison the next one.
    time.sleep(1.2)
    assert b.hit == 0


def test_zero_disables_it(budget_cls: type) -> None:
    """The escape hatch for a run that would rather hang than lose a complex."""
    b = budget_cls(0)
    b.arm()
    time.sleep(0.05)
    b.disarm()
    assert b.hit == 0

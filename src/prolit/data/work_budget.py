"""A wall-clock budget for one unit of corpus work, so one molecule costs one.

Corpus builders walk tens of thousands of complexes in a single process, and a
single pathological molecule can wedge the whole run: RDKit's
``GetSubstructMatches`` over a highly symmetric ligand explores a factorial
space, and ``maxMatches`` caps the results it keeps, not the search it does.
One decoy shard spun a core for eleven hours that way and produced nothing
after its first six.

Two builders need this -- the decoy corpus and the stapled CrossDocked corpus
-- so it lives here rather than in either of them.
"""

from __future__ import annotations

import signal

__all__ = ["WorkBudget"]


class WorkBudget:
    """Abandon one unit of work rather than the shard it is in.

    SIGALRM, not a process pool, because the loops that use this are already
    single-threaded inside a worker and a pool would restructure them. The
    limit of that choice is worth being explicit about: a Python signal handler
    runs between bytecodes, so a call that stays inside C for hours is **not**
    interrupted by this. It catches the interruptible majority; checkpointing
    ``meta.json`` as the build goes is what covers the rest, by making a wedged
    shard's finished work readable anyway.

    The raised :class:`TimeoutError` is an ``OSError``, so a per-unit
    ``except Exception`` already skips the offending item -- this only has to
    make the clock run.

    ``seconds <= 0`` disables it: the escape hatch for a run that would rather
    hang than lose an item.
    """

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self.hit = 0
        if seconds > 0:
            signal.signal(signal.SIGALRM, self._raise)

    def arm(self) -> None:
        if self.seconds > 0:
            signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def disarm(self) -> None:
        if self.seconds > 0:
            signal.setitimer(signal.ITIMER_REAL, 0)

    def _raise(self, _signum: int, _frame: object) -> None:
        self.hit += 1
        msg = f"work unit exceeded its {self.seconds}s budget"
        raise TimeoutError(msg)

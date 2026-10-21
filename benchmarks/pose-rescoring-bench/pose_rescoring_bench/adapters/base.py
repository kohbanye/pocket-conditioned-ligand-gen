"""Common interface every pose-rescoring baseline implements.

A baseline turns a list of CASF-2016 docking-power targets into this bench's
per-pose dump: ``pdbid, pose, rmsd, head``. It is scored on the SAME decoy stacks
and the SAME RMSD reference as our arms -- that is what makes the table a
comparison, and it is also why a reproduced number differs slightly from the
paper's, which used its own inputs.

``head`` is the raw model output. It is NOT oriented here: every consumer passes
a dump through :func:`pose_rescoring_bench.metrics.rescoring.orient`, which flips
the sign when the column correlates positively with RMSD. Orienting twice would
cancel out, so an adapter writes what the model said and nothing more.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

#: How many missing targets to name before truncating the error message.
_MAX_LISTED = 10

#: The schema every adapter returns, a subset of ``io_dumps.POSE_COLUMNS``
#: (baselines produce no ``pll``).
DUMP_COLUMNS = ("pdbid", "pose", "rmsd", "head")


class PoseBaseline(ABC):
    """Score CASF decoy poses with a published model, in that model's own env.

    Subclasses set ``name``, implement :meth:`score`, and do any heavy loading or
    environment checking in :meth:`setup`. Each runs a driver script under a
    dedicated interpreter (``paths.*_PYTHON``) because their dependencies cannot
    share an environment.
    """

    name: str = "base"
    #: True when the model consumes explicit protein-ligand contact or
    #: interaction features. Recorded because it decides which comparisons are
    #: like-for-like: our tokenizer sees no hand-specified contacts.
    uses_explicit_contacts: bool = False
    #: True when the model is a physics-based empirical function rather than one
    #: fitted to a training set, which changes what a leakage argument means.
    is_empirical: bool = False

    def setup(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Check sources, weights and environment. Raise with a fixable message."""

    @abstractmethod
    def score(self, targets: Sequence[str]) -> pd.DataFrame:
        """Per-pose predictions in :data:`DUMP_COLUMNS` order.

        A target the model cannot process must be reported, never dropped in
        silence: a partial dump scored as if complete is how a method gets credit
        for skipping the hard cases. Docking power is a per-target average, so a
        dropped target does not even show up as a lower score.
        """

    def teardown(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Release resources. Optional."""

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def check_coverage(
        frame: pd.DataFrame,
        targets: Sequence[str],
        allow_missing: Sequence[str] = (),
    ) -> None:
        """Fail loudly when a dump does not cover every requested target.

        ``allow_missing`` names targets the method provably cannot process --
        GenScore and RTMScore both fail to build a pocket graph for ``4k18``
        from the distributed inputs, and the numbers published for them were
        computed on the remaining 284. Naming those targets is how that becomes
        a reportable fact rather than a silent shortfall: the allowance has to
        be typed on the command line, and what it excused ends up in the run's
        log next to the numbers.
        """
        got = set(frame["pdbid"])
        want = set(targets) - set(allow_missing)
        missing = sorted(want - got)
        if missing:
            shown = missing[:_MAX_LISTED]
            more = "..." if len(missing) > _MAX_LISTED else ""
            msg = (
                f"{len(missing)} of {len(want)} targets are absent from the "
                f"dump: {shown}{more}. "
                "Report them rather than comparing on a reduced set."
            )
            raise ValueError(msg)

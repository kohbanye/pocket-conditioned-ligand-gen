"""Common interface every affinity baseline implements.

A baseline turns a list of complexes into this bench's per-complex dump:
``pdbid, logka, cluster, head``. It is scored on the SAME crystal poses and the
SAME labels as our arms -- that is what makes the table a comparison, and it is
also why a reproduced number differs slightly from the paper's, which used its
own inputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from affinity_bench.inference import Complex

#: How many missing ids to name before truncating the error message.
_MAX_LISTED = 10

#: The schema every adapter returns, matching affinity_bench.io_dumps.
DUMP_COLUMNS = ("pdbid", "logka", "cluster", "head")


class AffinityBaseline(ABC):
    """Score complexes with a published model, in that model's own environment.

    Subclasses set ``name``, implement :meth:`score`, and do any heavy loading
    in :meth:`setup`. Most run a driver script under a dedicated interpreter
    (``paths.*_PYTHON``) because their dependencies cannot share an environment.
    """

    name: str = "base"
    #: True when the model consumes explicit protein-ligand contact or
    #: interaction features. Recorded because it decides which comparisons are
    #: like-for-like: our tokenizer sees no hand-specified contacts.
    uses_explicit_contacts: bool = False

    def setup(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Check weights and environment. Raise with a fixable message."""

    @abstractmethod
    def score(self, complexes: Sequence[Complex]) -> pd.DataFrame:
        """Per-complex predictions in :data:`DUMP_COLUMNS` order.

        A complex the model cannot process must be reported, never dropped in
        silence: a partial dump scored as if complete is how a method gets
        credit for skipping the hard cases.
        """

    def teardown(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Release resources. Optional."""

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def check_coverage(frame: pd.DataFrame, complexes: Sequence[Complex]) -> None:
        """Fail loudly when a dump does not cover every requested complex."""
        got = set(frame["pdbid"])
        want = {c.pdbid for c in complexes}
        missing = sorted(want - got)
        if missing:
            shown = missing[:_MAX_LISTED]
            more = "..." if len(missing) > _MAX_LISTED else ""
            msg = (
                f"{len(missing)} of {len(want)} complexes are absent from the "
                f"dump: {shown}{more}. "
                "Report them rather than comparing on a reduced set."
            )
            raise ValueError(msg)

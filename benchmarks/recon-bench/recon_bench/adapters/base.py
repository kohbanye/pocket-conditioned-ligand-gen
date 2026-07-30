"""Common interface every reconstruction adapter implements."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterable

from recon_bench.types import ReconResult, Sample


class ReconstructionModel(ABC):
    """Encode a 3D structure to discrete tokens and decode it back.

    Subclasses set ``name`` and the capability flags, then implement
    :meth:`reconstruct`. Heavy resources (weights, child interpreters) are
    created in :meth:`setup`, which the runner calls once before the loop.
    """

    name: str = "base"
    can_protein: bool = False
    can_ligand: bool = False

    def setup(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Load weights / prepare child processes. Idempotent."""

    @abstractmethod
    def reconstruct(self, sample: Sample) -> ReconResult:
        """Reconstruct one sample. Must not raise: on failure return a
        ``ReconResult`` with ``ok=False`` and an ``error`` message."""

    def reconstruct_batch(self, samples: Iterable[Sample]) -> list[ReconResult]:
        """Reconstruct many samples. Default is a serial loop; subclasses that
        benefit from batching (e.g. one subprocess call) override this."""
        return [self._timed(s) for s in samples]

    def teardown(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Release resources. Optional."""

    # -- helpers ----------------------------------------------------------
    def _timed(self, sample: Sample) -> ReconResult:
        t0 = time.perf_counter()
        try:
            result = self.reconstruct(sample)
        except Exception as exc:  # noqa: BLE001 - adapters must never crash the run
            result = ReconResult(
                model=self.name, sample_id=sample.sample_id, ok=False, error=repr(exc)
            )
        if result.runtime_s is None:
            result.runtime_s = time.perf_counter() - t0
        return result

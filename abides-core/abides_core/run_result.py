"""Lean, timing-only result returned from :meth:`Kernel.terminate`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from . import NanosecondTime


@dataclass(frozen=True, slots=True)
class KernelRunResult:
    """Scheduling facts for one completed kernel run.

    Attributes:
      elapsed: Wall-clock time spent inside :meth:`Kernel.runner`.
      slowest_agent_finish_time: The maximum
        ``Kernel._agent_current_times`` value at termination — the
        latest "agent-busy-until" timestamp observed.
      messages_processed: Total number of messages dequeued during
        the run.
    """

    elapsed: timedelta
    slowest_agent_finish_time: NanosecondTime
    messages_processed: int

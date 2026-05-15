"""Pluggable observer protocol for kernel-side metric collection.

The kernel never aggregates domain metrics itself. Callers register
zero or more :class:`KernelObserver` instances at construction; the
kernel and agents dispatch lifecycle events to them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .kernel import Kernel


@runtime_checkable
class KernelObserver(Protocol):
    """Receives metric reports and the terminal kernel event.

    All methods are called synchronously from the kernel / agent event
    loop. Implementations must not block.
    """

    def on_metric(self, agent_id: int, agent_type: str, key: str, value: float) -> None:
        """Called once per :meth:`Agent.report_metric` call."""

    def on_terminate(self, kernel: Kernel) -> None:
        """Called from :meth:`Kernel.terminate` after all
        ``kernel_terminating()`` callbacks have run.
        """


class DefaultMetricsObserver:
    """Reproduces the v2 mean-by-type metrics print on terminate.

    Aggregates ``(agent_type, key) -> (sum, count)`` and emits one
    log line per ``(type, key)`` from :meth:`on_terminate`.
    """

    def __init__(self) -> None:
        self._aggregates: dict[str, dict[str, dict[str, float]]] = {}

    def on_metric(self, agent_id: int, agent_type: str, key: str, value: float) -> None:
        type_bucket = self._aggregates.setdefault(agent_type, {})
        agg = type_bucket.setdefault(key, {"sum": 0.0, "count": 0.0})
        agg["sum"] += float(value)
        agg["count"] += 1.0

    def on_terminate(self, kernel: Kernel) -> None:
        import logging

        logger = logging.getLogger("abides_core.kernel")
        if not self._aggregates:
            return
        logger.info("Mean reported metrics by agent type:")
        for agent_type, metrics in self._aggregates.items():
            for key, agg in metrics.items():
                count = agg.get("count", 0.0)
                if not count:
                    continue
                mean = agg["sum"] / count
                logger.info(f"{agent_type}.{key}: {mean:.6g} (n={int(count)})")

    @property
    def aggregates(self) -> dict[str, dict[str, dict[str, float]]]:
        """Read-only access to the accumulated ``type -> key -> {sum,
        count}`` mapping. Useful for tests and downstream tooling.
        """
        return self._aggregates

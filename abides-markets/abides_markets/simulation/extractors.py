"""ResultExtractor — protocol and helpers for custom simulation result extraction.

Custom extractors let callers inject domain-specific logic into
``_extract_result()`` without modifying the library.  The extracted value is
stored in ``SimulationResult.extensions[extractor.key]``.

Usage
-----
Using a function::

    from abides_markets.simulation import FunctionExtractor, run_simulation

    my_extractor = FunctionExtractor(
        key="exchange_stream",
        fn=lambda result, agents: agents[0].stream_history,
    )
    result = run_simulation(config, extractors=[my_extractor])
    data = result.extensions["exchange_stream"]

Subclassing for more complex logic::

    from abides_markets.simulation import BaseResultExtractor

    class MyExtractor(BaseResultExtractor):
        key = "my_metric"

        def extract(self, result, agents):
            # ... compute something ...
            return 42.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, runtime_checkable

from typing_extensions import Protocol

from abides_core.agent import Agent
from abides_core.run_result import KernelRunResult


@runtime_checkable
class ResultExtractor(Protocol):
    """Protocol satisfied by any object with a ``key`` attribute and ``extract`` method.

    Structural sub-typing: you do not need to inherit from this class.
    """

    key: str
    """Unique key under which the extracted value is stored in ``SimulationResult.extensions``."""

    def extract(self, result: KernelRunResult, agents: list[Agent]) -> Any:
        """Extract a value from the kernel run output.

        Parameters
        ----------
        result:
            Scheduling facts (elapsed wall clock, message count,
            slowest agent finish time) returned by
            ``Kernel.terminate()``.
        agents:
            The kernel's agent list, in id order, with all log buffers
            populated.

        Returns
        -------
        Any
            Any picklable value.  For REST compatibility prefer JSON-native types
            (dicts, lists, ints, floats, strings).
        """
        ...


class BaseResultExtractor(ABC):
    """Base class for extractors that need state or complex initialisation.

    Subclass this and implement :meth:`extract`.  The ``key`` class attribute
    must be set on the concrete subclass.
    """

    key: str

    @abstractmethod
    def extract(self, result: KernelRunResult, agents: list[Agent]) -> Any:
        """See :class:`ResultExtractor.extract`."""


class FunctionExtractor:
    """Convenience wrapper that turns a plain callable into a :class:`ResultExtractor`.

    Parameters
    ----------
    key:
        The key under which the result is stored in ``SimulationResult.extensions``.
    fn:
        A callable ``(result, agents) -> Any``.

    Example
    -------
    ::

        FunctionExtractor("n_messages", lambda r, a: r.messages_processed)
    """

    def __init__(
        self,
        key: str,
        fn: Callable[[KernelRunResult, list[Agent]], Any],
    ) -> None:
        self.key = key
        self._fn = fn

    def extract(self, result: KernelRunResult, agents: list[Agent]) -> Any:
        return self._fn(result, agents)

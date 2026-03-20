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
        fn=lambda end_state: end_state["agents"][0].stream_history,
    )
    result = run_simulation(config, extractors=[my_extractor])
    data = result.extensions["exchange_stream"]

Subclassing for more complex logic::

    from abides_markets.simulation import BaseResultExtractor

    class MyExtractor(BaseResultExtractor):
        key = "my_metric"

        def extract(self, end_state: dict) -> float:
            agents = end_state["agents"]
            # ... compute something ...
            return 42.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, runtime_checkable

from typing_extensions import Protocol


@runtime_checkable
class ResultExtractor(Protocol):
    """Protocol satisfied by any object with a ``key`` attribute and ``extract`` method.

    Structural sub-typing: you do not need to inherit from this class.
    """

    key: str
    """Unique key under which the extracted value is stored in ``SimulationResult.extensions``."""

    def extract(self, end_state: dict[str, Any]) -> Any:
        """Extract a value from the raw ABIDES ``end_state`` dict.

        Parameters
        ----------
        end_state:
            The dictionary returned by ``Kernel.terminate()`` — contains the
            ``"agents"`` list plus any entries added via ``Kernel.custom_state``.

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
    def extract(self, end_state: dict[str, Any]) -> Any:
        """See :class:`ResultExtractor.extract`."""


class FunctionExtractor:
    """Convenience wrapper that turns a plain callable into a :class:`ResultExtractor`.

    Parameters
    ----------
    key:
        The key under which the result is stored in ``SimulationResult.extensions``.
    fn:
        A callable ``(end_state: dict) -> Any``.

    Example
    -------
    ::

        FunctionExtractor("n_messages", lambda e: e.get("ttl_messages", 0))
    """

    def __init__(self, key: str, fn: Callable[[dict[str, Any]], Any]) -> None:
        self.key = key
        self._fn = fn

    def extract(self, end_state: dict[str, Any]) -> Any:
        return self._fn(end_state)

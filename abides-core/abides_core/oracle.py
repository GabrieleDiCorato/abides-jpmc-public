"""Structural Protocol for oracles consumed by the kernel and agents.

The kernel core does not import any concrete oracle. Any object that
matches this Protocol (e.g. the concrete ``abides_markets.oracles.Oracle``
ABC and its subclasses) is accepted by ``Kernel(oracle=...)``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from . import NanosecondTime


@runtime_checkable
class Oracle(Protocol):
    """Minimal contract a market-data oracle must satisfy.

    Concrete oracles in :mod:`abides_markets` already conform to this
    Protocol structurally and need no inheritance change.
    """

    f_log: dict[str, list[dict[str, Any]]]

    def get_daily_open_price(
        self, symbol: str, mkt_open: NanosecondTime, cents: bool = True
    ) -> int: ...

    def observe_price(
        self,
        symbol: str,
        current_time: NanosecondTime,
        random_state: np.random.RandomState,
        sigma_n: int = 1000,
    ) -> int: ...

"""Shared test fixtures for abides-markets.

Provides helpers that eliminate per-test boilerplate for constructing agents
with stubbed kernel state and for building quick simulation configs.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from abides_core import NanosecondTime
from abides_core.utils import datetime_str_to_ns, str_to_ns
from abides_markets.config_system import SimulationBuilder

# ---------------------------------------------------------------------------
# Shared time constants
# ---------------------------------------------------------------------------

DATE: NanosecondTime = datetime_str_to_ns("20210205")
MKT_OPEN: NanosecondTime = DATE + str_to_ns("09:30:00")
MKT_CLOSE: NanosecondTime = MKT_OPEN + str_to_ns("06:30:00")


# ---------------------------------------------------------------------------
# Agent construction helper
# ---------------------------------------------------------------------------


class StubKernel:
    """Minimal kernel stand-in so agents can be constructed outside a simulation."""

    oracle = None

    def send_message(self, sender_id: int, recipient_id: int, message: Any) -> None:
        pass  # swallow all outgoing messages


def make_agent(
    agent_cls: type,
    *,
    seed: int = 42,
    symbol: str = "TEST",
    starting_cash: int = 10_000_000,
    exchange_id: int = 99,
    mkt_open: NanosecondTime = MKT_OPEN,
    mkt_close: NanosecondTime = MKT_CLOSE,
    current_time: NanosecondTime | None = None,
    **kwargs: Any,
) -> Any:
    """Construct an agent with kernel stubs already wired.

    Handles the boilerplate that every unit test needs:
    ``random_state``, ``kernel``, ``exchange_id``, ``mkt_open / mkt_close``,
    and ``current_time``.

    Pass any extra constructor kwargs (e.g. ``r_bar=100_000`` for ValueAgent)
    via ``**kwargs``.

    Example::

        agent = make_agent(NoiseAgent, wakeup_time=MKT_OPEN + str_to_ns("00:05:00"))
        assert agent.state == "AWAITING_WAKEUP"
    """
    defaults: dict[str, Any] = {
        "id": 0,
        "random_state": np.random.RandomState(seed),
        "symbol": symbol,
        "starting_cash": starting_cash,
    }
    # Merge: caller kwargs win over defaults.
    merged = {**defaults, **kwargs}

    agent = agent_cls(**merged)

    # Wire kernel stubs.
    agent.kernel = StubKernel()
    agent.exchange_id = exchange_id
    agent.mkt_open = mkt_open
    agent.mkt_close = mkt_close
    agent.current_time = current_time or (mkt_open + str_to_ns("00:05:00"))

    return agent


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stub_kernel() -> StubKernel:
    """A fresh StubKernel instance."""
    return StubKernel()


@pytest.fixture(scope="session")
def rmsc04_config():
    """A minimal rmsc04 config (2-min sim) reusable across tests."""
    return (
        SimulationBuilder()
        .from_template("rmsc04")
        .market(end_time="09:32:00")
        .seed(42)
        .build()
    )

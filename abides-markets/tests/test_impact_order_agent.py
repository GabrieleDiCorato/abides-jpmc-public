"""Tests for ImpactOrderAgent and ImpactOrderAgentConfig.

Covers:
- Constructor validation (invalid order_type, quantity < 1, LIMIT without price)
- State machine: AWAITING_WAKEUP → DONE (MARKET / LIMIT)
- State machine: AWAITING_WAKEUP → AWAITING_SPREAD → DONE (AGGRESSIVE_LIMIT)
- Early wakeup rescheduling (current_time < order_time)
- DONE state ignores subsequent wakeups / messages
- _submit_order: MARKET, LIMIT, AGGRESSIVE_LIMIT with populated book
- _submit_order: AGGRESSIVE_LIMIT with empty book → market-order fallback + warning
- Config validation (pydantic): LIMIT without limit_price, offset after close
- Config system: agent creation, field propagation, side conversion, name/type tags
- Registration: "impact_order" present in registry with correct metadata
- Builder integration: enable_agent("impact_order", ...) round-trips via compile
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from abides_core.utils import str_to_ns
from abides_markets.agents.impact_order_agent import ImpactOrderAgent
from abides_markets.orders import Side

# ---------------------------------------------------------------------------
# Shared time constants (must match conftest.py so make_agent() is consistent)
# ---------------------------------------------------------------------------
from tests.conftest import MKT_CLOSE, MKT_OPEN, make_agent

ORDER_TIME: int = MKT_OPEN + str_to_ns("01:00:00")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_impact_agent(**kwargs: Any) -> ImpactOrderAgent:
    """Construct an ImpactOrderAgent with kernel stubs wired."""
    defaults: dict[str, Any] = dict(
        order_time=ORDER_TIME,
        quantity=100,
    )
    defaults.update(kwargs)
    return make_agent(ImpactOrderAgent, **defaults)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# 1. Constructor validation
# ---------------------------------------------------------------------------


class TestImpactOrderAgentConstructor:
    def test_invalid_order_type_raises(self):
        with pytest.raises(ValueError, match="invalid order_type"):
            _make_impact_agent(order_type="UNKNOWN")

    def test_quantity_zero_raises(self):
        with pytest.raises(ValueError, match="quantity must be ≥ 1"):
            _make_impact_agent(quantity=0)

    def test_quantity_negative_raises(self):
        with pytest.raises(ValueError, match="quantity must be ≥ 1"):
            _make_impact_agent(quantity=-5)

    def test_limit_without_price_raises(self):
        with pytest.raises(ValueError, match="limit_price is required"):
            _make_impact_agent(order_type="LIMIT", limit_price=None)

    def test_limit_with_price_ok(self):
        agent = _make_impact_agent(order_type="LIMIT", limit_price=10_000)
        assert agent.limit_price == 10_000

    def test_defaults(self):
        agent = _make_impact_agent()
        assert agent.order_type == "MARKET"
        assert agent.side == Side.BID
        assert agent.quantity == 100
        assert agent.state == "AWAITING_WAKEUP"

    def test_valid_states_declared(self):
        assert "AWAITING_WAKEUP" in ImpactOrderAgent.VALID_STATES
        assert "AWAITING_SPREAD" in ImpactOrderAgent.VALID_STATES
        assert "DONE" in ImpactOrderAgent.VALID_STATES

    def test_starting_cash_default(self):
        # Default is $1 000 000 (100_000_000 cents) — enough for large blocks.
        agent = _make_impact_agent(starting_cash=100_000_000)
        assert agent.starting_cash == 100_000_000


# ---------------------------------------------------------------------------
# 2. wakeup() — early arrival (before order_time)
# ---------------------------------------------------------------------------


class TestWakeupEarlyArrival:
    def test_early_wakeup_reschedules(self):
        """If woken before order_time the agent must reschedule at order_time."""
        agent = _make_impact_agent()
        # Simulate being awoken 30 min before the order_time.
        early = ORDER_TIME - str_to_ns("30min")
        agent.current_time = early

        with patch.object(agent, "set_wakeup") as mock_wakeup:
            # Patch super().wakeup to return True (market is open).
            with patch.object(type(agent).__mro__[1], "wakeup", return_value=True):
                agent.wakeup(early)
            mock_wakeup.assert_called_once_with(ORDER_TIME)

        assert agent.state == "AWAITING_WAKEUP"

    def test_early_wakeup_does_not_submit(self):
        agent = _make_impact_agent()
        early = ORDER_TIME - str_to_ns("30min")
        agent.current_time = early

        with (
            patch.object(agent, "_submit_order") as mock_submit,
            patch.object(agent, "set_wakeup"),
            patch.object(type(agent).__mro__[1], "wakeup", return_value=True),
        ):
            agent.wakeup(early)
        mock_submit.assert_not_called()


# ---------------------------------------------------------------------------
# 3. wakeup() — on time (MARKET and LIMIT)
# ---------------------------------------------------------------------------


class TestWakeupOnTimeDirect:
    """MARKET and LIMIT submit immediately in wakeup(); no spread query needed."""

    @pytest.mark.parametrize("order_type", ["MARKET", "LIMIT"])
    def test_submits_immediately_on_time(self, order_type: str):
        kwargs: dict[str, Any] = dict(order_type=order_type)
        if order_type == "LIMIT":
            kwargs["limit_price"] = 10_000
        agent = _make_impact_agent(**kwargs)
        agent.current_time = ORDER_TIME

        with patch.object(agent, "_submit_order") as mock_submit:
            with patch.object(type(agent).__mro__[1], "wakeup", return_value=True):
                agent.wakeup(ORDER_TIME)
            mock_submit.assert_called_once()

        assert agent.state == "DONE"

    def test_market_does_not_query_spread(self):
        agent = _make_impact_agent(order_type="MARKET")
        agent.current_time = ORDER_TIME

        with (
            patch.object(agent, "get_current_spread") as mock_spread,
            patch.object(agent, "_submit_order"),
            patch.object(type(agent).__mro__[1], "wakeup", return_value=True),
        ):
            agent.wakeup(ORDER_TIME)
        mock_spread.assert_not_called()


# ---------------------------------------------------------------------------
# 4. wakeup() — AGGRESSIVE_LIMIT queries spread first
# ---------------------------------------------------------------------------


class TestWakeupAggressiveLimit:
    def test_queries_spread_on_time(self):
        agent = _make_impact_agent(order_type="AGGRESSIVE_LIMIT")
        agent.current_time = ORDER_TIME

        with patch.object(agent, "get_current_spread") as mock_spread:
            with patch.object(type(agent).__mro__[1], "wakeup", return_value=True):
                agent.wakeup(ORDER_TIME)
            mock_spread.assert_called_once_with(agent.symbol)

        assert agent.state == "AWAITING_SPREAD"

    def test_does_not_submit_until_spread_arrives(self):
        agent = _make_impact_agent(order_type="AGGRESSIVE_LIMIT")
        agent.current_time = ORDER_TIME

        with (
            patch.object(agent, "_submit_order") as mock_submit,
            patch.object(agent, "get_current_spread"),
            patch.object(type(agent).__mro__[1], "wakeup", return_value=True),
        ):
            agent.wakeup(ORDER_TIME)
        mock_submit.assert_not_called()


# ---------------------------------------------------------------------------
# 5. wakeup() — market closed guard
# ---------------------------------------------------------------------------


class TestWakeupMarketClosed:
    def test_market_closed_returns_early(self):
        """super().wakeup() returning False must abort all logic."""
        agent = _make_impact_agent()
        agent.current_time = ORDER_TIME

        with patch.object(agent, "_submit_order") as mock_submit:
            with patch.object(type(agent).__mro__[1], "wakeup", return_value=False):
                agent.wakeup(ORDER_TIME)
            mock_submit.assert_not_called()

        assert agent.state == "AWAITING_WAKEUP"


# ---------------------------------------------------------------------------
# 6. wakeup() — DONE state is a no-op
# ---------------------------------------------------------------------------


class TestWakeupDoneIgnored:
    def test_subsequent_wakeup_after_done_does_nothing(self):
        agent = _make_impact_agent()
        agent.state = "DONE"
        agent.current_time = ORDER_TIME

        with patch.object(agent, "_submit_order") as mock_submit:
            with patch.object(type(agent).__mro__[1], "wakeup", return_value=True):
                agent.wakeup(ORDER_TIME + str_to_ns("1min"))
            mock_submit.assert_not_called()

        assert agent.state == "DONE"


# ---------------------------------------------------------------------------
# 7. receive_message() — spread response triggers submission
# ---------------------------------------------------------------------------


class TestReceiveMessageSpreadResponse:
    def _spread_msg(self) -> Any:
        from abides_markets.messages.query import QuerySpreadResponseMsg

        return QuerySpreadResponseMsg(
            symbol="TEST",
            bids=[(10_000, 500)],
            asks=[(10_010, 500)],
            mkt_closed=False,
            depth=1,
            last_trade=None,
        )

    def test_spread_response_triggers_submit(self):
        agent = _make_impact_agent(order_type="AGGRESSIVE_LIMIT")
        agent.state = "AWAITING_SPREAD"
        agent.current_time = ORDER_TIME

        msg = self._spread_msg()

        with patch.object(agent, "_submit_order") as mock_submit:
            with patch.object(type(agent).__mro__[1], "receive_message"):
                agent.receive_message(ORDER_TIME, sender_id=99, message=msg)
            mock_submit.assert_called_once()

        assert agent.state == "DONE"

    def test_non_spread_message_ignored_in_awaiting_spread(self):
        """Other message types while AWAITING_SPREAD must not trigger submit."""
        from abides_markets.messages.market import MarketHoursMsg

        agent = _make_impact_agent(order_type="AGGRESSIVE_LIMIT")
        agent.state = "AWAITING_SPREAD"
        agent.current_time = ORDER_TIME

        other_msg = MarketHoursMsg(MKT_OPEN, MKT_CLOSE)

        with patch.object(agent, "_submit_order") as mock_submit:
            with patch.object(type(agent).__mro__[1], "receive_message"):
                agent.receive_message(ORDER_TIME, sender_id=99, message=other_msg)
            mock_submit.assert_not_called()

        assert agent.state == "AWAITING_SPREAD"

    def test_spread_response_ignored_when_done(self):
        """Spread arrival after DONE must not re-submit."""
        agent = _make_impact_agent(order_type="AGGRESSIVE_LIMIT")
        agent.state = "DONE"
        agent.current_time = ORDER_TIME

        msg = self._spread_msg()

        with patch.object(agent, "_submit_order") as mock_submit:
            with patch.object(type(agent).__mro__[1], "receive_message"):
                agent.receive_message(ORDER_TIME, sender_id=99, message=msg)
            mock_submit.assert_not_called()

        assert agent.state == "DONE"


# ---------------------------------------------------------------------------
# 8. _submit_order: individual order types
# ---------------------------------------------------------------------------


class TestSubmitOrder:
    def test_market_order_calls_place_market(self):
        agent = _make_impact_agent(order_type="MARKET", quantity=500)

        with patch.object(agent, "place_market_order") as mock_mo:
            agent._submit_order()
        mock_mo.assert_called_once_with(agent.symbol, 500, agent.side)

    def test_limit_order_calls_place_limit(self):
        agent = _make_impact_agent(order_type="LIMIT", quantity=200, limit_price=9_950)

        with patch.object(agent, "place_limit_order") as mock_lo:
            agent._submit_order()
        mock_lo.assert_called_once_with(agent.symbol, 200, agent.side, 9_950)

    def test_aggressive_limit_bid_uses_ask_price(self):
        """BID aggressive limit → price is the current ask."""
        agent = _make_impact_agent(
            order_type="AGGRESSIVE_LIMIT", side=Side.BID, quantity=100
        )
        # Inject known bid/ask into cache.
        agent.known_bids["TEST"] = [(9_990, 300)]
        agent.known_asks["TEST"] = [(10_010, 300)]

        with patch.object(agent, "place_limit_order") as mock_lo:
            agent._submit_order()
        mock_lo.assert_called_once_with(agent.symbol, 100, Side.BID, 10_010)

    def test_aggressive_limit_ask_uses_bid_price(self):
        """ASK aggressive limit → price is the current bid."""
        agent = _make_impact_agent(
            order_type="AGGRESSIVE_LIMIT", side=Side.ASK, quantity=100
        )
        agent.known_bids["TEST"] = [(9_990, 300)]
        agent.known_asks["TEST"] = [(10_010, 300)]

        with patch.object(agent, "place_limit_order") as mock_lo:
            agent._submit_order()
        mock_lo.assert_called_once_with(agent.symbol, 100, Side.ASK, 9_990)

    def test_aggressive_limit_empty_bid_book_falls_back_to_market(self):
        """AGGRESSIVE_LIMIT sell with empty bid side → market order + warning."""
        agent = _make_impact_agent(
            order_type="AGGRESSIVE_LIMIT", side=Side.ASK, quantity=100
        )
        # No bids in the book.
        agent.known_bids["TEST"] = []
        agent.known_asks["TEST"] = [(10_010, 300)]

        with (
            patch.object(agent, "place_market_order") as mock_mo,
            patch.object(agent, "place_limit_order") as mock_lo,
            pytest.warns(RuntimeWarning, match="book side is empty"),
        ):
            agent._submit_order()
        mock_mo.assert_called_once_with(agent.symbol, 100, Side.ASK)
        mock_lo.assert_not_called()

    def test_aggressive_limit_empty_ask_book_falls_back_to_market(self):
        """AGGRESSIVE_LIMIT buy with empty ask side → market order + warning."""
        agent = _make_impact_agent(
            order_type="AGGRESSIVE_LIMIT", side=Side.BID, quantity=100
        )
        agent.known_bids["TEST"] = [(9_990, 300)]
        agent.known_asks["TEST"] = []

        with (
            patch.object(agent, "place_market_order") as mock_mo,
            pytest.warns(RuntimeWarning, match="book side is empty"),
        ):
            agent._submit_order()
        mock_mo.assert_called_once_with(agent.symbol, 100, Side.BID)

    def test_aggressive_limit_completely_empty_book_falls_back_to_market(self):
        """Both sides empty → fallback (bid agent using ask price → None)."""
        agent = _make_impact_agent(
            order_type="AGGRESSIVE_LIMIT", side=Side.BID, quantity=100
        )
        agent.known_bids["TEST"] = []
        agent.known_asks["TEST"] = []

        with (
            patch.object(agent, "place_market_order") as mock_mo,
            pytest.warns(RuntimeWarning),
        ):
            agent._submit_order()
        mock_mo.assert_called_once()


# ---------------------------------------------------------------------------
# 9. Side: ASK agent
# ---------------------------------------------------------------------------


class TestAskSide:
    def test_ask_agent_defaults(self):
        agent = _make_impact_agent(side=Side.ASK, order_type="MARKET")
        assert agent.side == Side.ASK

    def test_market_order_uses_ask_side(self):
        agent = _make_impact_agent(side=Side.ASK, order_type="MARKET", quantity=300)
        with patch.object(agent, "place_market_order") as mock_mo:
            agent._submit_order()
        mock_mo.assert_called_once_with(agent.symbol, 300, Side.ASK)


# ---------------------------------------------------------------------------
# 10. Config validation (pydantic)
# ---------------------------------------------------------------------------


class TestImpactOrderAgentConfig:
    def _ctx(self):
        from abides_markets.config_system.agent_configs import AgentCreationContext

        return AgentCreationContext(
            ticker="TEST",
            mkt_open=MKT_OPEN,
            mkt_close=MKT_CLOSE,
            log_orders=False,
            oracle_r_bar=None,
        )

    def test_limit_without_price_raises_validation_error(self):
        from pydantic import ValidationError

        from abides_markets.config_system.agent_configs import ImpactOrderAgentConfig

        with pytest.raises(ValidationError, match="limit_price"):
            ImpactOrderAgentConfig(
                order_time_offset="01:00:00",
                quantity=100,
                order_type="LIMIT",
                limit_price=None,
            )

    def test_limit_with_price_valid(self):
        from abides_markets.config_system.agent_configs import ImpactOrderAgentConfig

        cfg = ImpactOrderAgentConfig(
            order_time_offset="01:00:00",
            quantity=100,
            order_type="LIMIT",
            limit_price=10_000,
        )
        assert cfg.limit_price == 10_000

    def test_offset_after_close_raises(self):
        """order_time_offset that places order at/after mkt_close must be rejected."""
        from abides_markets.config_system.agent_configs import ImpactOrderAgentConfig

        # Market session is 6h30m; 7h offset goes beyond close.
        cfg = ImpactOrderAgentConfig(
            order_time_offset="07:00:00",
            quantity=100,
            order_type="MARKET",
        )
        with pytest.raises(ValueError, match="after market close"):
            cfg.create_agents(
                count=1,
                id_start=0,
                master_rng=np.random.RandomState(1),
                context=self._ctx(),
            )

    def test_creates_agent_with_correct_order_time(self):
        from abides_markets.config_system.agent_configs import ImpactOrderAgentConfig

        cfg = ImpactOrderAgentConfig(
            order_time_offset="01:00:00",
            quantity=250,
            order_type="MARKET",
        )
        agents = cfg.create_agents(
            count=1,
            id_start=0,
            master_rng=np.random.RandomState(1),
            context=self._ctx(),
        )
        assert len(agents) == 1
        agent = agents[0]
        assert isinstance(agent, ImpactOrderAgent)
        expected_order_time = MKT_OPEN + str_to_ns("01:00:00")
        assert agent.order_time == expected_order_time

    def test_creates_agent_quantity_propagated(self):
        from abides_markets.config_system.agent_configs import ImpactOrderAgentConfig

        cfg = ImpactOrderAgentConfig(
            order_time_offset="00:30:00",
            quantity=5_000,
            order_type="MARKET",
        )
        agents = cfg.create_agents(
            count=1,
            id_start=0,
            master_rng=np.random.RandomState(1),
            context=self._ctx(),
        )
        assert agents[0].quantity == 5_000

    def test_creates_agent_side_bid(self):
        from abides_markets.config_system.agent_configs import ImpactOrderAgentConfig

        cfg = ImpactOrderAgentConfig(
            order_time_offset="01:00:00",
            quantity=100,
            side="BID",
        )
        agents = cfg.create_agents(
            count=1,
            id_start=0,
            master_rng=np.random.RandomState(1),
            context=self._ctx(),
        )
        assert agents[0].side == Side.BID

    def test_creates_agent_side_ask(self):
        from abides_markets.config_system.agent_configs import ImpactOrderAgentConfig

        cfg = ImpactOrderAgentConfig(
            order_time_offset="01:00:00",
            quantity=100,
            side="ASK",
        )
        agents = cfg.create_agents(
            count=1,
            id_start=0,
            master_rng=np.random.RandomState(1),
            context=self._ctx(),
        )
        assert agents[0].side == Side.ASK

    def test_creates_agent_limit_price_propagated(self):
        from abides_markets.config_system.agent_configs import ImpactOrderAgentConfig

        cfg = ImpactOrderAgentConfig(
            order_time_offset="01:00:00",
            quantity=100,
            order_type="LIMIT",
            limit_price=10_500,
        )
        agents = cfg.create_agents(
            count=1,
            id_start=0,
            master_rng=np.random.RandomState(1),
            context=self._ctx(),
        )
        assert agents[0].limit_price == 10_500

    def test_name_and_type_tags_set(self):
        from abides_markets.config_system.agent_configs import ImpactOrderAgentConfig

        cfg = ImpactOrderAgentConfig(
            order_time_offset="01:00:00",
            quantity=100,
        )
        agents = cfg.create_agents(
            count=1,
            id_start=7,
            master_rng=np.random.RandomState(1),
            context=self._ctx(),
        )
        assert agents[0].name == "ImpactOrderAgent_7"
        assert agents[0].type == "ImpactOrderAgent"

    def test_multiple_agents_get_sequential_names(self):
        from abides_markets.config_system.agent_configs import ImpactOrderAgentConfig

        cfg = ImpactOrderAgentConfig(
            order_time_offset="01:00:00",
            quantity=100,
        )
        agents = cfg.create_agents(
            count=3,
            id_start=10,
            master_rng=np.random.RandomState(1),
            context=self._ctx(),
        )
        assert agents[0].name == "ImpactOrderAgent_10"
        assert agents[1].name == "ImpactOrderAgent_11"
        assert agents[2].name == "ImpactOrderAgent_12"

    def test_extra_fields_rejected(self):
        """Config model has extra='forbid' — unknown fields must be rejected."""
        from pydantic import ValidationError

        from abides_markets.config_system.agent_configs import ImpactOrderAgentConfig

        with pytest.raises(ValidationError):
            ImpactOrderAgentConfig(
                order_time_offset="01:00:00",
                quantity=100,
                nonexistent_field="bad",
            )

    def test_missing_required_fields_raises(self):
        """order_time_offset and quantity are required — omitting either fails."""
        from pydantic import ValidationError

        from abides_markets.config_system.agent_configs import ImpactOrderAgentConfig

        with pytest.raises(ValidationError):
            ImpactOrderAgentConfig(quantity=100)  # missing order_time_offset

        with pytest.raises(ValidationError):
            ImpactOrderAgentConfig(order_time_offset="01:00:00")  # missing quantity


# ---------------------------------------------------------------------------
# 11. Registry
# ---------------------------------------------------------------------------


class TestImpactOrderRegistration:
    def test_registered_in_registry(self):
        from abides_markets.config_system.registry import registry

        entry = registry.get("impact_order")
        assert entry is not None
        assert entry.name == "impact_order"

    def test_category_is_execution(self):
        from abides_markets.config_system.registry import registry

        entry = registry.get("impact_order")
        assert entry.category == "execution"

    def test_does_not_require_oracle(self):
        from abides_markets.config_system.registry import registry

        entry = registry.get("impact_order")
        assert entry.requires_oracle is False

    def test_typical_count_range(self):
        from abides_markets.config_system.registry import registry

        entry = registry.get("impact_order")
        assert entry.typical_count_range == (1, 1)

    def test_recommended_with_contains_noise(self):
        from abides_markets.config_system.registry import registry

        entry = registry.get("impact_order")
        assert "noise" in entry.recommended_with


# ---------------------------------------------------------------------------
# 12. Builder / compiler integration
# ---------------------------------------------------------------------------


class TestImpactOrderBuilderIntegration:
    def test_enable_agent_accepted_by_builder(self):
        from abides_markets.config_system import SimulationBuilder

        config = (
            SimulationBuilder()
            .market(
                oracle={"type": "sparse_mean_reverting"},
                date="20210205",
                end_time="16:00:00",
            )
            .enable_agent("noise", count=10)
            .enable_agent(
                "impact_order",
                count=1,
                order_time_offset="01:00:00",
                quantity=1_000,
                order_type="MARKET",
            )
            .seed(42)
            .build()
        )
        assert "impact_order" in config.agents
        assert config.agents["impact_order"].enabled is True
        assert config.agents["impact_order"].params["quantity"] == 1_000

    def test_compile_creates_impact_agent_instance(self):
        from abides_markets.config_system import SimulationBuilder, compile

        config = (
            SimulationBuilder()
            .market(
                oracle={"type": "sparse_mean_reverting"},
                date="20210205",
                end_time="16:00:00",
            )
            .enable_agent("noise", count=5)
            .enable_agent(
                "impact_order",
                count=1,
                order_time_offset="01:00:00",
                quantity=500,
            )
            .seed(42)
            .build()
        )
        runtime = compile(config)
        impact_agents = [
            a for a in runtime["agents"] if isinstance(a, ImpactOrderAgent)
        ]
        assert len(impact_agents) == 1
        assert impact_agents[0].quantity == 500

    def test_compile_side_and_order_type_propagated(self):
        from abides_markets.config_system import SimulationBuilder, compile

        config = (
            SimulationBuilder()
            .market(
                oracle={"type": "sparse_mean_reverting"},
                date="20210205",
                end_time="16:00:00",
            )
            .enable_agent("noise", count=5)
            .enable_agent(
                "impact_order",
                count=1,
                order_time_offset="00:30:00",
                quantity=200,
                side="ASK",
                order_type="AGGRESSIVE_LIMIT",
            )
            .seed(42)
            .build()
        )
        runtime = compile(config)
        impact = next(a for a in runtime["agents"] if isinstance(a, ImpactOrderAgent))
        assert impact.side == Side.ASK
        assert impact.order_type == "AGGRESSIVE_LIMIT"

    def test_compile_limit_with_price_propagated(self):
        from abides_markets.config_system import SimulationBuilder, compile

        config = (
            SimulationBuilder()
            .market(
                oracle={"type": "sparse_mean_reverting"},
                date="20210205",
                end_time="16:00:00",
            )
            .enable_agent("noise", count=5)
            .enable_agent(
                "impact_order",
                count=1,
                order_time_offset="01:00:00",
                quantity=300,
                order_type="LIMIT",
                limit_price=9_800,
            )
            .seed(42)
            .build()
        )
        runtime = compile(config)
        impact = next(a for a in runtime["agents"] if isinstance(a, ImpactOrderAgent))
        assert impact.limit_price == 9_800
        assert impact.order_type == "LIMIT"

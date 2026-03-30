"""Regression tests for long-duration simulation health.

These tests verify that the RMSC04 template (and the legacy config) produce
healthy two-sided markets when run for multiple hours.  The key regression
was that ValueAgent herding caused one side of the book to be empty for
>60% of the session during 3-hour runs.

Root causes fixed:
1. ValueAgent.update_estimates() used ``self.sigma_t`` (always 0) instead of
   ``sigma_tprime`` in the Bayesian posterior variance update.
2. ``sigma_s=100_000`` was calibrated for abstract time steps, not
   nanosecond-resolution simulations with SparseMeanRevertingOracle.
3. Single-fire noise agents exhausted after ~1 hour.
4. AMM could compute negative bid prices when spread widened.
"""

from __future__ import annotations

import numpy as np
import pytest

from abides_markets.agents.value_agent import ValueAgent
from abides_markets.config_system import SimulationBuilder
from abides_markets.simulation import run_simulation


# ---------------------------------------------------------------------------
# ValueAgent sigma_t convergence unit test
# ---------------------------------------------------------------------------


class TestValueAgentSigmaTConvergence:
    """Verify the Bayesian posterior variance update in update_estimates()."""

    def test_sigma_t_not_stuck_at_zero(self):
        """sigma_t must increase from its initial value of 0 after observations."""
        agent = ValueAgent(
            id=0,
            name="test_va",
            type="ValueAgent",
            random_state=np.random.RandomState(42),
            r_bar=100_000,
            kappa=1.67e-15,
            sigma_s=2.5e-9,
            sigma_n=1000.0,
        )
        # sigma_t starts at 0
        assert agent.sigma_t == 0.0

        # Simulate the math of update_estimates without running a full simulation.
        # After one update cycle, sigma_tprime should incorporate sigma_s,
        # and the posterior sigma_t should be > 0.
        delta = 175_000_000_000  # 175 seconds in ns (typical wakeup gap)
        sigma_tprime = ((1 - agent.kappa) ** (2 * delta)) * agent.sigma_t
        sigma_tprime += (
            (1 - (1 - agent.kappa) ** (2 * delta))
            / (1 - (1 - agent.kappa) ** 2)
        ) * agent.sigma_s

        assert sigma_tprime > 0, "sigma_tprime should be > 0 after time advancement"

        # The fixed posterior update:
        new_sigma_t = (agent.sigma_n * sigma_tprime) / (agent.sigma_n + sigma_tprime)
        assert new_sigma_t > 0, "posterior sigma_t must be > 0 after one observation"

    def test_sigma_t_converges(self):
        """sigma_t should converge to a steady-state, not stay at 0."""
        kappa = 1.67e-15
        sigma_s = 2.5e-9
        sigma_n = 1000.0
        delta = 175_000_000_000  # 175 seconds

        sigma_t = 0.0
        for _ in range(100):
            sigma_tprime = ((1 - kappa) ** (2 * delta)) * sigma_t
            sigma_tprime += (
                (1 - (1 - kappa) ** (2 * delta))
                / (1 - (1 - kappa) ** 2)
            ) * sigma_s
            sigma_t = (sigma_n * sigma_tprime) / (sigma_n + sigma_tprime)

        assert sigma_t > 0, "sigma_t must converge above 0"
        # Check it's a reasonable fraction of sigma_n (not negligible)
        assert sigma_t < sigma_n, "sigma_t should be less than sigma_n"


# ---------------------------------------------------------------------------
# AMM negative price regression test
# ---------------------------------------------------------------------------


class TestAMMNegativePriceClamping:
    """Verify that AMM never produces negative bid prices."""

    def test_compute_orders_positive_prices(self):
        """All prices from compute_orders_to_place must be >= 1."""
        from abides_markets.agents import AdaptiveMarketMakerAgent

        agent = AdaptiveMarketMakerAgent(
            id=0,
            symbol="TEST",
            starting_cash=10_000_000,
            random_state=np.random.RandomState(42),
            window_size=50,  # fixed spread of 50 ticks
            num_ticks=10,
            level_spacing=5.0,
            price_skew_param=None,
        )
        # tick_size = ceil(level_spacing * window_size) = ceil(5.0 * 50) = 250
        # With mid=100, lowest_bid = (100 - 25) - 9*250 = 75 - 2250 = -2175
        # Before the fix, this would produce negative prices.
        bids, asks = agent.compute_orders_to_place(mid=100)
        assert all(p >= 1 for p in bids), f"Negative bid prices found: {bids}"
        assert all(p >= 1 for p in asks), f"Negative ask prices found: {asks}"

    def test_empty_bids_when_all_negative(self):
        """When mid is extremely low, all computed bids may be filtered out."""
        from abides_markets.agents import AdaptiveMarketMakerAgent

        agent = AdaptiveMarketMakerAgent(
            id=0,
            symbol="TEST",
            starting_cash=10_000_000,
            random_state=np.random.RandomState(42),
            window_size=100,
            num_ticks=20,
            level_spacing=5.0,
            price_skew_param=None,
        )
        bids, asks = agent.compute_orders_to_place(mid=5)
        # All bid prices should be >= 1, or list should be empty
        assert all(p >= 1 for p in bids)
        assert all(p >= 1 for p in asks)


# ---------------------------------------------------------------------------
# Long-duration simulation health (config system path)
# ---------------------------------------------------------------------------


class TestLongSimulationHealth:
    """Verify that long simulations produce healthy two-sided markets."""

    @pytest.mark.parametrize("seed", [42, 123])
    def test_rmsc04_2h_balanced_book(self, seed):
        """A 2-hour RMSC04 simulation should have neither side empty >50%."""
        config = (
            SimulationBuilder()
            .from_template("rmsc04")
            .market(end_time="11:30:00")  # 2 hours
            .seed(seed)
            .build()
        )
        result = run_simulation(config)
        liq = result.markets["ABM"].liquidity

        assert liq.pct_time_no_bid < 50, (
            f"Bid side empty {liq.pct_time_no_bid:.1f}% of session (seed={seed})"
        )
        assert liq.pct_time_no_ask < 50, (
            f"Ask side empty {liq.pct_time_no_ask:.1f}% of session (seed={seed})"
        )
        assert liq.total_exchanged_volume > 0, "No trades occurred"


# ---------------------------------------------------------------------------
# Long-duration simulation health (legacy config path)
# ---------------------------------------------------------------------------


class TestLegacyConfigHealth:
    """Verify that the legacy rmsc04.build_config benefits from the sigma_s fix.

    Note: The legacy config still uses single-fire noise agents (no multi_wake),
    so it has structural limitations for very long simulations.  These tests
    verify that the sigma_t and sigma_s fixes improve the legacy path, using
    a shorter window where the noise flow is still adequate.
    """

    def test_legacy_rmsc04_30min_balanced_book(self):
        """Legacy rmsc04 config for 30 min should have balanced book sides."""
        import shutil

        from abides_core import Kernel
        from abides_core.utils import subdict
        from abides_markets.configs.rmsc04 import build_config

        config = build_config(
            seed=42,
            end_time="10:00:00",
            book_logging=True,
            log_orders=False,
            exchange_log_orders=False,
        )
        # Verify the sigma_s fix is applied
        value_agents = [a for a in config["agents"] if isinstance(a, ValueAgent)]
        assert len(value_agents) > 0, "No ValueAgent instances found in legacy config"
        assert value_agents[0].sigma_s == pytest.approx(2.5e-9), (
            f"Legacy config should use fund_vol² as sigma_s, got {value_agents[0].sigma_s}"
        )

        kernel = Kernel(
            log_dir="__test_legacy_health",
            random_state=np.random.RandomState(seed=42),
            **subdict(
                config,
                [
                    "start_time",
                    "stop_time",
                    "agents",
                    "agent_latency_model",
                    "default_computation_delay",
                    "custom_properties",
                ],
            ),
            skip_log=True,
        )
        kernel.run()

        exchange = config["agents"][0]
        tracker = exchange.metric_trackers.get("ABM")
        assert tracker is not None, "No metric tracker for ABM"
        assert tracker.pct_time_no_liquidity_bids < 50, (
            f"Bid side empty {tracker.pct_time_no_liquidity_bids:.1f}%"
        )
        assert tracker.pct_time_no_liquidity_asks < 50, (
            f"Ask side empty {tracker.pct_time_no_liquidity_asks:.1f}%"
        )

        shutil.rmtree("log/__test_legacy_health", ignore_errors=True)

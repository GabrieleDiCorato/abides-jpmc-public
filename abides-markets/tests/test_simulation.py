"""Tests for abides_markets.simulation package.

Structure
---------
* TestResultProfile  — flag composition, containment
* TestSchemas        — pandera schema validation (valid + invalid inputs)
* TestResultModels   — Pydantic model construction, immutability, serialisation
* TestExtractors     — FunctionExtractor, BaseResultExtractor protocol check
* TestRunSimulation  — end-to-end integration against a short rmsc04 run
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from abides_markets.config_system import SimulationBuilder
from abides_markets.simulation import (
    AgentData,
    BaseResultExtractor,
    FunctionExtractor,
    L1Close,
    L1DataFrameSchema,
    L1Snapshots,
    L2DataFrameSchema,
    L2Snapshots,
    LiquidityMetrics,
    MarketSummary,
    RawLogsSchema,
    ResultExtractor,
    ResultProfile,
    SimulationMetadata,
    SimulationResult,
    run_simulation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def short_config():
    """A minimal rmsc04-based config that finishes in seconds."""
    return (
        SimulationBuilder()
        .from_template("rmsc04")
        .market(end_time="09:32:00")  # 2-minute sim
        .seed(42)
        .build()
    )


@pytest.fixture(scope="session")
def summary_result(short_config, tmp_path_factory):
    """Run once per session and return a SUMMARY-profile result."""
    log_dir = str(tmp_path_factory.mktemp("sim_logs"))
    return run_simulation(short_config, profile=ResultProfile.SUMMARY, log_dir=log_dir)


@pytest.fixture(scope="session")
def full_result(short_config, tmp_path_factory):
    """Run once per session and return a FULL-profile result."""
    log_dir = str(tmp_path_factory.mktemp("sim_logs_full"))
    return run_simulation(short_config, profile=ResultProfile.FULL, log_dir=log_dir)


# ---------------------------------------------------------------------------
# TestResultProfile
# ---------------------------------------------------------------------------


class TestResultProfile:
    def test_summary_contains_base_flags(self):
        assert ResultProfile.METADATA in ResultProfile.SUMMARY
        assert ResultProfile.AGENT_PNL in ResultProfile.SUMMARY
        assert ResultProfile.LIQUIDITY in ResultProfile.SUMMARY

    def test_summary_excludes_heavy_flags(self):
        assert ResultProfile.L1_SERIES not in ResultProfile.SUMMARY
        assert ResultProfile.L2_SERIES not in ResultProfile.SUMMARY
        assert ResultProfile.AGENT_LOGS not in ResultProfile.SUMMARY

    def test_quant_contains_summary(self):
        for flag in (
            ResultProfile.METADATA,
            ResultProfile.AGENT_PNL,
            ResultProfile.LIQUIDITY,
        ):
            assert flag in ResultProfile.QUANT

    def test_quant_contains_series_flags(self):
        assert ResultProfile.L1_SERIES in ResultProfile.QUANT
        assert ResultProfile.L2_SERIES in ResultProfile.QUANT

    def test_full_contains_all(self):
        for flag in (
            ResultProfile.METADATA,
            ResultProfile.AGENT_PNL,
            ResultProfile.LIQUIDITY,
            ResultProfile.L1_SERIES,
            ResultProfile.L2_SERIES,
            ResultProfile.AGENT_LOGS,
        ):
            assert flag in ResultProfile.FULL

    def test_composition(self):
        custom = ResultProfile.L1_SERIES | ResultProfile.METADATA
        assert ResultProfile.L1_SERIES in custom
        assert ResultProfile.METADATA in custom
        assert ResultProfile.L2_SERIES not in custom


# ---------------------------------------------------------------------------
# TestSchemas
# ---------------------------------------------------------------------------


class TestL1DataFrameSchema:
    def _valid_df(self):
        return pd.DataFrame(
            {
                "time_ns": pd.array([1_000_000, 2_000_000], dtype="Int64"),
                "bid_price_cents": pd.array([9990, 9985], dtype="Int64"),
                "bid_qty": pd.array([100, 200], dtype="Int64"),
                "ask_price_cents": pd.array([10010, 10020], dtype="Int64"),
                "ask_qty": pd.array([150, 300], dtype="Int64"),
            }
        )

    def test_valid_passes(self):
        df = self._valid_df()
        validated = L1DataFrameSchema.validate(df)
        assert len(validated) == 2

    def test_nullable_fields_allowed(self):
        df = self._valid_df()
        df.loc[0, "bid_price_cents"] = pd.NA
        df.loc[0, "bid_qty"] = pd.NA
        L1DataFrameSchema.validate(df)  # should not raise

    def test_extra_column_rejected(self):
        df = self._valid_df()
        df["extra"] = 1
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            L1DataFrameSchema.validate(df)

    def test_negative_time_rejected(self):
        df = self._valid_df()
        df.loc[0, "time_ns"] = -1
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            L1DataFrameSchema.validate(df)


class TestL2DataFrameSchema:
    def _valid_df(self):
        return pd.DataFrame(
            {
                "time_ns": pd.array([100, 100, 200], dtype="Int64"),
                "side": ["bid", "ask", "bid"],
                "level": pd.array([0, 0, 0], dtype="Int64"),
                "price_cents": pd.array([9990, 10010, 9988], dtype="Int64"),
                "qty": pd.array([100, 150, 50], dtype="Int64"),
            }
        )

    def test_valid_passes(self):
        L2DataFrameSchema.validate(self._valid_df())

    def test_zero_price_rejected(self):
        df = self._valid_df()
        df.loc[0, "price_cents"] = 0
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            L2DataFrameSchema.validate(df)

    def test_zero_qty_rejected(self):
        df = self._valid_df()
        df.loc[1, "qty"] = 0
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            L2DataFrameSchema.validate(df)

    def test_invalid_side_rejected(self):
        df = self._valid_df()
        df.loc[0, "side"] = "unknown"
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            L2DataFrameSchema.validate(df)

    def test_extra_column_rejected(self):
        df = self._valid_df()
        df["extra"] = 1
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            L2DataFrameSchema.validate(df)


class TestRawLogsSchema:
    def test_extra_columns_retained(self):
        """strict=False means extra columns pass through unchanged."""
        df = pd.DataFrame(
            {
                "EventTime": pd.array([1000], dtype="Int64"),
                "EventType": ["ORDER_SUBMITTED"],
                "agent_id": pd.array([1], dtype="Int64"),
                "agent_type": ["NoiseAgent"],
                "symbol": ["ABM"],  # extra column — should pass through
            }
        )
        validated = RawLogsSchema.validate(df)
        assert "symbol" in validated.columns


# ---------------------------------------------------------------------------
# TestResultModels
# ---------------------------------------------------------------------------


def _make_metadata():
    return SimulationMetadata(
        seed=42,
        tickers=["ABM"],
        sim_start_ns=1_000_000_000,
        sim_end_ns=5_000_000_000,
        wall_clock_elapsed_s=1.23,
        config_snapshot={"ticker": "ABM"},
    )


def _make_market_summary():
    return MarketSummary(
        symbol="ABM",
        l1_close=L1Close(
            time_ns=4_999_999_999, bid_price_cents=9990, ask_price_cents=10010
        ),
        liquidity=LiquidityMetrics(
            pct_time_no_bid=1.0,
            pct_time_no_ask=0.5,
            total_exchanged_volume=5000,
            last_trade_cents=10000,
            vwap_cents=None,
        ),
    )


def _make_agent_data():
    return AgentData(
        agent_id=1,
        agent_type="NoiseAgent",
        agent_name="noise_agent_1",
        agent_category="background",
        final_holdings={"CASH": 100_000_000, "ABM": 10},
        starting_cash_cents=100_000_000,
        mark_to_market_cents=100_100_000,
        pnl_cents=100_000,
        pnl_pct=0.1,
    )


def _make_simulation_result(include_logs: bool = False) -> SimulationResult:
    logs = None
    profile = ResultProfile.SUMMARY
    if include_logs:
        logs = pd.DataFrame(
            {
                "EventTime": pd.array([1000, 2000], dtype="Int64"),
                "EventType": ["ORDER_SUBMITTED", "ORDER_EXECUTED"],
                "agent_id": pd.array([1, 1], dtype="Int64"),
                "agent_type": ["NoiseAgent", "NoiseAgent"],
                "symbol": ["ABM", "ABM"],
                "order_id": pd.array([100, 100], dtype="Int64"),
                "quantity": pd.array([10, 10], dtype="Int64"),
                "side": ["BID", "BID"],
                "fill_price": pd.array([pd.NA, 10000], dtype="Int64"),
                "limit_price": pd.array([10010, 10010], dtype="Int64"),
            }
        )
        profile = ResultProfile.FULL

    return SimulationResult(
        metadata=_make_metadata(),
        markets={"ABM": _make_market_summary()},
        agents=[_make_agent_data()],
        logs=logs,
        profile=profile,
    )


class TestGetAgentsByCategory:
    def test_single_match(self):
        result = _make_simulation_result()
        # Default helper creates a "background" agent
        assert len(result.get_agents_by_category("background")) == 1
        assert result.get_agents_by_category("background")[0].agent_id == 1

    def test_no_match(self):
        result = _make_simulation_result()
        assert result.get_agents_by_category("strategy") == []

    def test_multiple_categories(self):
        bg = AgentData(
            agent_id=1,
            agent_type="NoiseAgent",
            agent_name="noise_1",
            agent_category="background",
            final_holdings={"CASH": 100_000_000},
            starting_cash_cents=100_000_000,
            mark_to_market_cents=100_000_000,
            pnl_cents=0,
            pnl_pct=0.0,
        )
        strat = AgentData(
            agent_id=2,
            agent_type="MeanReversionAgent",
            agent_name="mr_1",
            agent_category="strategy",
            final_holdings={"CASH": 100_000_000},
            starting_cash_cents=100_000_000,
            mark_to_market_cents=100_100_000,
            pnl_cents=100_000,
            pnl_pct=0.1,
        )
        result = SimulationResult(
            metadata=_make_metadata(),
            markets={"ABM": _make_market_summary()},
            agents=[bg, strat],
            profile=ResultProfile.SUMMARY,
        )
        assert len(result.get_agents_by_category("strategy")) == 1
        assert result.get_agents_by_category("strategy")[0].agent_id == 2
        assert len(result.get_agents_by_category("background")) == 1


class TestSimulationResultImmutability:
    def test_frozen_prevents_field_reassignment(self):
        from pydantic import ValidationError

        result = _make_simulation_result()
        with pytest.raises(ValidationError):
            result.metadata = _make_metadata()

    def test_frozen_nested_model(self):
        from pydantic import ValidationError

        result = _make_simulation_result()
        with pytest.raises(ValidationError):
            result.metadata.seed = 99


class TestL1SnapshotsReadOnly:
    def test_numpy_arrays_not_writeable(self):
        snaps = L1Snapshots(
            times_ns=np.array([1, 2, 3], dtype=np.int64),
            bid_prices=np.array([9990, 9985, None], dtype=object),
            bid_quantities=np.array([100, 200, None], dtype=object),
            ask_prices=np.array([10010, 10020, None], dtype=object),
            ask_quantities=np.array([150, 300, None], dtype=object),
        )
        assert not snaps.times_ns.flags.writeable
        with pytest.raises(ValueError):
            snaps.times_ns[0] = 999

    def test_as_dataframe_validates(self):
        snaps = L1Snapshots(
            times_ns=np.array([1_000_000], dtype=np.int64),
            bid_prices=np.array([9990], dtype=object),
            bid_quantities=np.array([100], dtype=object),
            ask_prices=np.array([10010], dtype=object),
            ask_quantities=np.array([150], dtype=object),
        )
        df = snaps.as_dataframe()
        assert list(df.columns) == [
            "time_ns",
            "bid_price_cents",
            "bid_qty",
            "ask_price_cents",
            "ask_qty",
        ]
        assert len(df) == 1


class TestL2SnapshotsReadOnly:
    def test_times_ns_not_writeable(self):
        snaps = L2Snapshots(
            times_ns=np.array([1, 2], dtype=np.int64),
            bids=[[(9990, 100), (9980, 50)], [(9990, 80)]],
            asks=[[(10010, 150)], [(10010, 120), (10020, 30)]],
        )
        assert not snaps.times_ns.flags.writeable

    def test_as_dataframe_long_format(self):
        snaps = L2Snapshots(
            times_ns=np.array([1000], dtype=np.int64),
            bids=[[(9990, 100), (9980, 50)]],
            asks=[[(10010, 150)]],
        )
        df = snaps.as_dataframe()
        assert set(df.columns) == {"time_ns", "side", "level", "price_cents", "qty"}
        # 2 bid levels + 1 ask level = 3 rows
        assert len(df) == 3
        assert set(df["side"].unique()) == {"bid", "ask"}
        # All prices must be > 0
        assert (df["price_cents"] > 0).all()

    def test_as_dataframe_validates_no_zero_price(self):
        """Confirm schema rejects any zero-padded snapshot that sneaks in."""
        snaps = L2Snapshots(
            times_ns=np.array([1000], dtype=np.int64),
            bids=[[(0, 0)]],  # invalid — should fail schema
            asks=[[(10010, 150)]],
        )
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            snaps.as_dataframe()


class TestSimulationResultSerialization:
    def test_to_json_roundtrip(self):
        result = _make_simulation_result()
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["metadata"]["seed"] == 42
        assert "ABM" in parsed["markets"]

    def test_to_dict_no_numpy(self):
        result = _make_simulation_result()
        d = result.to_dict()

        # Walk through the dict recursively and confirm no numpy/DataFrame objects
        def _check(obj):
            assert not isinstance(
                obj, (np.ndarray, pd.DataFrame)
            ), f"Found non-JSON-native type: {type(obj)}"
            if isinstance(obj, dict):
                for v in obj.values():
                    _check(v)
            elif isinstance(obj, list):
                for v in obj:
                    _check(v)

        _check(d)

    def test_to_json_is_valid_json(self):
        result = _make_simulation_result()
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)


class TestSimulationResultSummary:
    def test_summary_always_works(self):
        result = _make_simulation_result()
        s = result.summary()
        assert "ABM" in s
        assert "seed=42" in s

    def test_summary_includes_pnl(self):
        result = _make_simulation_result()
        s = result.summary()
        assert "NoiseAgent" in s


class TestOrderLogs:
    def test_order_logs_filters_correctly(self):
        result = _make_simulation_result(include_logs=True)
        df = result.order_logs()
        assert set(df["EventType"].unique()).issubset(
            {
                "ORDER_SUBMITTED",
                "ORDER_ACCEPTED",
                "ORDER_EXECUTED",
                "ORDER_CANCELLED",
                "PARTIAL_CANCELLED",
                "ORDER_MODIFIED",
                "ORDER_REPLACED",
            }
        )

    def test_order_logs_raises_without_logs_profile(self):
        result = _make_simulation_result(include_logs=False)
        with pytest.raises(RuntimeError, match="Agent logs were not extracted"):
            result.order_logs()


# ---------------------------------------------------------------------------
# TestExtractors
# ---------------------------------------------------------------------------


class TestExtractors:
    def test_function_extractor_satisfies_protocol(self):
        ext = FunctionExtractor("key", lambda r, a: 42)
        assert isinstance(ext, ResultExtractor)

    def test_function_extractor_calls_fn(self):
        ext = FunctionExtractor("n_agents", lambda r, a: len(a))
        result = ext.extract(None, [object(), object()])
        assert result == 2

    def test_base_extractor_satisfies_protocol(self):
        class MyExt(BaseResultExtractor):
            key = "my_key"

            def extract(self, result, agents):
                return "value"

        ext = MyExt()
        assert isinstance(ext, ResultExtractor)
        assert ext.extract(None, []) == "value"


# ---------------------------------------------------------------------------
# TestRunSimulation — integration (slow, session-scoped fixtures)
# ---------------------------------------------------------------------------


class TestRunSimulationSummary:
    def test_result_is_simulation_result(self, summary_result):
        assert isinstance(summary_result, SimulationResult)

    def test_metadata_ticker(self, summary_result):
        assert "ABM" in summary_result.metadata.tickers

    def test_metadata_seed(self, summary_result):
        assert summary_result.metadata.seed == 42

    def test_markets_has_symbol(self, summary_result):
        assert "ABM" in summary_result.markets

    def test_l1_close_populated(self, summary_result):
        close = summary_result.markets["ABM"].l1_close
        # At least one side should have a price after a 2-min sim
        assert close.bid_price_cents is not None or close.ask_price_cents is not None

    def test_liquidity_volume_positive(self, summary_result):
        liq = summary_result.markets["ABM"].liquidity
        assert liq.total_exchanged_volume >= 0

    def test_agents_populated(self, summary_result):
        assert len(summary_result.agents) > 0

    def test_pnl_equals_mtm_minus_starting(self, summary_result):
        for agent in summary_result.agents:
            expected = agent.mark_to_market_cents - agent.starting_cash_cents
            assert agent.pnl_cents == expected

    def test_no_l1_series_in_summary(self, summary_result):
        assert summary_result.markets["ABM"].l1_series is None

    def test_no_l2_series_in_summary(self, summary_result):
        assert summary_result.markets["ABM"].l2_series is None

    def test_logs_none_in_summary(self, summary_result):
        assert summary_result.logs is None

    def test_to_json_roundtrip(self, summary_result):
        parsed = json.loads(summary_result.to_json())
        assert parsed["metadata"]["seed"] == 42


class TestRunSimulationFull:
    def test_logs_dataframe_not_none(self, full_result):
        assert full_result.logs is not None

    def test_logs_has_required_columns(self, full_result):
        df = full_result.logs
        assert "EventTime" in df.columns
        assert "EventType" in df.columns
        assert "agent_id" in df.columns
        assert "agent_type" in df.columns

    def test_order_logs_accessible(self, full_result):
        df = full_result.order_logs()
        assert len(df) > 0


class TestRunSimulationQuant:
    @pytest.fixture(scope="class")
    def quant_result(self, short_config, tmp_path_factory):
        log_dir = str(tmp_path_factory.mktemp("sim_logs_quant"))
        return run_simulation(
            short_config, profile=ResultProfile.QUANT, log_dir=log_dir
        )

    def test_l1_series_populated(self, quant_result):
        assert quant_result.markets["ABM"].l1_series is not None

    def test_l2_series_populated(self, quant_result):
        assert quant_result.markets["ABM"].l2_series is not None

    def test_l1_as_dataframe_validates(self, quant_result):
        df = quant_result.markets["ABM"].l1_series.as_dataframe()
        assert len(df) > 0

    def test_l2_as_dataframe_no_zero_prices(self, quant_result):
        df = quant_result.markets["ABM"].l2_series.as_dataframe()
        if len(df) > 0:
            assert (df["price_cents"] > 0).all()
            assert (df["qty"] > 0).all()

    def test_l1_times_read_only(self, quant_result):
        times = quant_result.markets["ABM"].l1_series.times_ns
        assert not times.flags.writeable

    def test_l2_times_read_only(self, quant_result):
        times = quant_result.markets["ABM"].l2_series.times_ns
        assert not times.flags.writeable


class TestRunSimulationExtractor:
    def test_function_extractor_injected(self, short_config, tmp_path_factory):
        ext = FunctionExtractor("agent_count", lambda r, a: len(a))
        log_dir = str(tmp_path_factory.mktemp("sim_logs_ext"))
        result = run_simulation(short_config, extractors=[ext], log_dir=log_dir)
        assert "agent_count" in result.extensions
        assert isinstance(result.extensions["agent_count"], int)
        assert result.extensions["agent_count"] > 0

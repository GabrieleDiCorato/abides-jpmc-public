"""Comprehensive integration tests for pandas 2.2.x compatibility.

These tests go beyond unit tests to exercise real simulation paths,
log serialization, order book DataFrame operations, config date arithmetic,
and the get_value_from_timestamp fix.
"""

import datetime
import os
import tempfile

import numpy as np
import pandas as pd

from abides_core import abides
from abides_core.utils import (
    datetime_str_to_ns,
    fmt_ts,
    ns_date,
    parse_logs_df,
    str_to_ns,
)


# ============================================================================
# Test 1: rmsc03 config builds and runs end-to-end
# ============================================================================
class TestRmsc03EndToEnd:
    """rmsc03 uses str_to_ns('10S') (uppercase S) and datetime_str_to_ns.
    This verifies the full config+simulation pipeline works."""

    def test_rmsc03_config_builds(self):
        from abides_markets.configs.rmsc03 import build_config

        config = build_config(
            seed=42,
            num_noise_agents=50,  # reduced for speed
            num_value_agents=10,
            num_momentum_agents=5,
        )
        assert "agents" in config
        assert "start_time" in config
        assert "stop_time" in config
        assert config["stop_time"] > config["start_time"]

    def test_rmsc03_simulation_runs(self):
        from abides_markets.configs.rmsc03 import build_config

        config = build_config(
            seed=42,
            num_noise_agents=50,
            num_value_agents=10,
            num_momentum_agents=5,
            start_time="09:30:00",
            end_time="09:35:00",  # very short for speed
        )
        config["skip_log"] = True
        result, agents = abides.run(config)
        assert result is not None
        assert len(agents) > 0


# ============================================================================
# Test 2: Order book DataFrame operations (get_l3_itch)
# ============================================================================
class TestOrderBookDataFrameOps:
    """The order book get_l3_itch() method does complex .loc[] assignments
    on DataFrames. Verify these pandas operations work on 2.2.x."""

    def _run_sim_with_book_logging(self):
        from abides_markets.configs.rmsc04 import build_config

        config = build_config(seed=42)
        config["skip_log"] = True
        _, agents = abides.run(config)
        # Find the exchange agent
        exchange = None
        for agent in agents:
            if hasattr(agent, "order_books"):
                exchange = agent
                break
        return exchange

    def test_order_book_history_exists(self):
        exchange = self._run_sim_with_book_logging()
        assert exchange is not None
        for _symbol, book in exchange.order_books.items():
            assert len(book.history) > 0, "Order book should have history entries"

    def test_get_l3_itch_produces_dataframe(self):
        exchange = self._run_sim_with_book_logging()
        for _symbol, book in exchange.order_books.items():
            if len(book.history) > 0:
                df = book.get_l3_itch()
                assert isinstance(df, pd.DataFrame)
                assert "timestamp" in df.columns
                assert "type" in df.columns
                assert "reference" in df.columns
                assert "side" in df.columns
                assert "shares" in df.columns
                assert "price" in df.columns
                assert len(df) > 0

    def test_l3_itch_types_correct(self):
        exchange = self._run_sim_with_book_logging()
        for _symbol, book in exchange.order_books.items():
            if len(book.history) > 0:
                df = book.get_l3_itch()
                # Verify the type replacement worked
                valid_types = {"ADD", "CANCEL", "DELETE", "EXECUTE", "REPLACE"}
                actual_types = set(df["type"].dropna().unique())
                assert actual_types.issubset(
                    valid_types
                ), f"Unexpected types: {actual_types - valid_types}"

    def test_l3_itch_side_replacement(self):
        exchange = self._run_sim_with_book_logging()
        for _symbol, book in exchange.order_books.items():
            if len(book.history) > 0:
                df = book.get_l3_itch()
                valid_sides = {"B", "S"}
                actual_sides = set(df["side"].dropna().unique())
                assert actual_sides.issubset(
                    valid_sides
                ), f"Unexpected sides: {actual_sides - valid_sides}"


# ============================================================================
# Test 3: Log serialization roundtrip (to_pickle / read_pickle)
# ============================================================================
class TestLogSerialization:
    """Agent logs are serialized with to_pickle(compression='bz2').
    Verify this works on pandas 2.2.x."""

    def test_pickle_roundtrip(self):
        # Create a DataFrame mimicking an agent log
        df = pd.DataFrame(
            {
                "EventTime": [
                    1_612_517_400_000_000_000 + i * 1_000_000_000 for i in range(10)
                ],
                "EventType": ["WAKE" if i % 2 == 0 else "ORDER" for i in range(10)],
                "agent_id": [1] * 10,
            }
        )
        df.set_index("EventTime", inplace=True)

        with tempfile.NamedTemporaryFile(suffix=".bz2", delete=False) as f:
            path = f.name

        try:
            df.to_pickle(path, compression="bz2")
            df_read = pd.read_pickle(path, compression="bz2")
            pd.testing.assert_frame_equal(df, df_read)
        finally:
            os.unlink(path)

    def test_pickle_with_mixed_types(self):
        """Agent logs contain mixed types including dicts in Event column."""
        df = pd.DataFrame(
            {
                "EventTime": [1_612_517_400_000_000_000, 1_612_517_401_000_000_000],
                "EventType": ["WAKE", "ORDER_SUBMITTED"],
                "Event": [{"msg": "wakeup"}, {"order_id": 42, "price": 100_000}],
            }
        )
        df.set_index("EventTime", inplace=True)

        with tempfile.NamedTemporaryFile(suffix=".bz2", delete=False) as f:
            path = f.name

        try:
            df.to_pickle(path, compression="bz2")
            df_read = pd.read_pickle(path, compression="bz2")
            pd.testing.assert_frame_equal(df, df_read)
        finally:
            os.unlink(path)


# ============================================================================
# Test 4: parse_logs_df works with simulation output
# ============================================================================
class TestParseLogsDf:
    """parse_logs_df extracts and concatenates agent logs into one DataFrame."""

    def test_parse_logs_from_simulation(self):
        from abides_markets.configs.rmsc04 import build_config

        config = build_config(seed=42)
        config["skip_log"] = True
        _, agents = abides.run(config)
        df = parse_logs_df(agents)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "EventTime" in df.columns
        assert "EventType" in df.columns
        assert "agent_id" in df.columns

    def test_parse_logs_event_times_are_ints(self):
        from abides_markets.configs.rmsc04 import build_config

        config = build_config(seed=42)
        config["skip_log"] = True
        _, agents = abides.run(config)
        df = parse_logs_df(agents)
        # EventTime should be integer nanoseconds, not Timestamps
        assert df["EventTime"].dtype in (np.int64, np.int32, int, object)


# ============================================================================
# Test 5: get_value_from_timestamp fix
# ============================================================================
class TestGetValueFromTimestamp:
    """get_value_from_timestamp was refactored from get_loc(method='nearest')
    to get_indexer(method='nearest'). Verify correctness."""

    def test_exact_match(self):
        from abides_markets.utils import get_value_from_timestamp

        idx = pd.DatetimeIndex(
            [
                "2021-01-01 10:00:00",
                "2021-01-01 10:05:00",
                "2021-01-01 10:10:00",
            ]
        )
        s = pd.Series([100, 200, 300], index=idx)
        ts = datetime.datetime(2021, 1, 1, 10, 5, 0)
        result = get_value_from_timestamp(s, ts)
        assert result == 200

    def test_nearest_before(self):
        from abides_markets.utils import get_value_from_timestamp

        idx = pd.DatetimeIndex(
            [
                "2021-01-01 10:00:00",
                "2021-01-01 10:10:00",
            ]
        )
        s = pd.Series([100, 200], index=idx)
        # 10:03 is closer to 10:00 than 10:10
        ts = datetime.datetime(2021, 1, 1, 10, 3, 0)
        result = get_value_from_timestamp(s, ts)
        assert result == 100

    def test_nearest_after(self):
        from abides_markets.utils import get_value_from_timestamp

        idx = pd.DatetimeIndex(
            [
                "2021-01-01 10:00:00",
                "2021-01-01 10:10:00",
            ]
        )
        s = pd.Series([100, 200], index=idx)
        # 10:07 is closer to 10:10 than 10:00
        ts = datetime.datetime(2021, 1, 1, 10, 7, 0)
        result = get_value_from_timestamp(s, ts)
        assert result == 200

    def test_with_duplicated_index(self):
        from abides_markets.utils import get_value_from_timestamp

        idx = pd.DatetimeIndex(
            [
                "2021-01-01 10:00:00",
                "2021-01-01 10:00:00",  # duplicate
                "2021-01-01 10:05:00",
            ]
        )
        s = pd.Series([100, 150, 200], index=idx)
        ts = datetime.datetime(2021, 1, 1, 10, 0, 0)
        # The function keeps last for duplicates
        result = get_value_from_timestamp(s, ts)
        assert result == 150

    def test_single_element(self):
        from abides_markets.utils import get_value_from_timestamp

        idx = pd.DatetimeIndex(["2021-01-01 10:00:00"])
        s = pd.Series([42], index=idx)
        ts = datetime.datetime(2021, 1, 1, 15, 0, 0)
        result = get_value_from_timestamp(s, ts)
        assert result == 42


# ============================================================================
# Test 6: DataFrame .loc[] assignment patterns from order_book.py
# ============================================================================
class TestDataFrameLocAssignments:
    """order_book.py uses .loc[mask, col] = value patterns extensively.
    Verify these work without SettingWithCopyWarning on 2.2.x."""

    def test_loc_mask_assignment(self):
        df = pd.DataFrame(
            {
                "type": ["LIMIT", "CANCEL", "EXEC", "LIMIT"],
                "side": ["BID", "ASK", "BID", "ASK"],
                "price": [100, 200, 150, 300],
            }
        )
        df.loc[df.type == "EXEC", "side"] = np.nan
        assert pd.isna(df.loc[2, "side"])
        assert df.loc[0, "side"] == "BID"

    def test_loc_mask_with_apply(self):
        df = pd.DataFrame(
            {
                "tag": ["normal", "auctionFill", "normal"],
                "metadata": [None, {"quantity": 10, "price": 100}, None],
                "quantity": [5, 0, 3],
            }
        )
        df.loc[df.tag == "auctionFill", "quantity"] = df.loc[
            df.tag == "auctionFill", "metadata"
        ].apply(lambda x: x["quantity"])
        assert df.loc[1, "quantity"] == 10

    def test_replace_on_string_column(self):
        df = pd.DataFrame(
            {
                "type": ["LIMIT", "CANCEL", "CANCEL_PARTIAL", "EXEC"],
                "side": ["BID", "ASK", "BID", "ASK"],
            }
        )
        df["type"] = df["type"].replace(
            {
                "LIMIT": "ADD",
                "CANCEL_PARTIAL": "CANCEL",
                "CANCEL": "DELETE",
                "EXEC": "EXECUTE",
            }
        )
        assert list(df["type"]) == ["ADD", "DELETE", "CANCEL", "EXECUTE"]


# ============================================================================
# Test 7: pd.concat behavior with agent logs
# ============================================================================
class TestPdConcat:
    def test_concat_mixed_column_dfs(self):
        """parse_logs_df concatenates DataFrames with potentially different columns."""
        df1 = pd.DataFrame(
            {
                "EventTime": [1, 2],
                "EventType": ["A", "B"],
                "agent_id": [0, 0],
                "price": [100, 200],
            }
        )
        df2 = pd.DataFrame(
            {
                "EventTime": [3, 4],
                "EventType": ["C", "D"],
                "agent_id": [1, 1],
                "volume": [50, 60],
            }
        )
        result = pd.concat([df1, df2])
        assert len(result) == 4
        assert "price" in result.columns
        assert "volume" in result.columns

    def test_concat_empty_and_nonempty(self):
        df1 = pd.DataFrame()
        df2 = pd.DataFrame({"EventTime": [1], "EventType": ["A"]})
        result = pd.concat([df1, df2])
        assert len(result) == 1


# ============================================================================
# Test 8: Gym environment compatibility
# ============================================================================
class TestGymEnvironments:
    """Gym envs create configs internally. Verify they build and step."""

    def test_markets_execution_env(self):
        import gymnasium as gym

        import abides_gym  # noqa: F401 - triggers env registration

        env = gym.make("markets-execution-v0")
        obs, info = env.reset()
        assert obs is not None
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        env.close()

    def test_markets_daily_investor_env(self):
        import gymnasium as gym

        import abides_gym  # noqa: F401 - triggers env registration

        env = gym.make("markets-daily_investor-v0")
        obs, info = env.reset()
        assert obs is not None
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        env.close()


# ============================================================================
# Test 9: Full simulation with log writing (to_pickle) and reading
# ============================================================================
class TestFullSimulationWithLogs:
    """Run a simulation that actually writes logs, then read them back.
    Uses a temp directory so we only read freshly-written pickles (not stale
    ones from a previous pandas version)."""

    def test_simulation_writes_and_reads_logs(self):
        from abides_markets.configs.rmsc04 import build_config

        config = build_config(seed=99)
        # Enable logging to a unique temp directory
        config["skip_log"] = False

        # Use a unique log dir to avoid reading stale pandas-3.0 pickles
        import time

        unique_log_dir = f"integration_test_{int(time.time())}"

        abides.run(config, log_dir=unique_log_dir)

        # Check that log files were created and are readable
        log_path = os.path.join(".", "log", unique_log_dir)
        assert os.path.exists(log_path), f"Log directory {log_path} not created"

        bz2_files = [f for f in os.listdir(log_path) if f.endswith(".bz2")]
        assert len(bz2_files) > 0, "No .bz2 log files written"

        for f in bz2_files:
            filepath = os.path.join(log_path, f)
            df = pd.read_pickle(filepath, compression="bz2")
            assert isinstance(df, pd.DataFrame)

        # Cleanup
        import shutil

        shutil.rmtree(log_path, ignore_errors=True)


# ============================================================================
# Test 10: Timestamp edge cases
# ============================================================================
class TestTimestampEdgeCases:
    """Edge cases for the timestamp conversion functions."""

    def test_str_to_ns_with_hms_format(self):
        """'09:30:00' format used in all configs."""
        result = str_to_ns("09:30:00")
        assert result == 9 * 3_600_000_000_000 + 30 * 60_000_000_000

    def test_str_to_ns_with_hms_format_2(self):
        """'16:00:00' format used in all configs."""
        result = str_to_ns("16:00:00")
        assert result == 16 * 3_600_000_000_000

    def test_date_plus_time_matches_datetime(self):
        """Verify that date + HH:MM:SS equals a full datetime string."""
        date_ns = datetime_str_to_ns("20210205")
        time_ns = str_to_ns("09:30:00")
        combined = date_ns + time_ns
        direct = datetime_str_to_ns("2021-02-05 09:30:00")
        assert combined == direct

    def test_ns_date_on_all_config_dates(self):
        """Verify ns_date works for dates used in configs."""
        for date_str in ["20200603", "20210205"]:
            date_ns = datetime_str_to_ns(date_str)
            assert ns_date(date_ns) == date_ns
            # Adding any time should still round back to midnight
            with_time = date_ns + str_to_ns("12:34:56")
            assert ns_date(with_time) == date_ns

    def test_fmt_ts_all_config_patterns(self):
        """Verify fmt_ts for patterns actually used in configs."""
        date_ns = datetime_str_to_ns("20200603")
        mkt_open = date_ns + str_to_ns("09:30:00")
        mkt_close = date_ns + str_to_ns("16:00:00")

        assert fmt_ts(mkt_open) == "2020-06-03 09:30:00"
        assert fmt_ts(mkt_close) == "2020-06-03 16:00:00"

"""Data provider protocols for injecting external fundamental value data into ABIDES.

This module defines the API contract that external data sources must implement
to feed historical (or generated) data into an ExternalDataOracle. Two protocols
are provided for different use cases:

- **BatchDataProvider**: Returns an entire pd.Series at once. Best for
  file-based or pre-computed data that fits comfortably in memory.
- **PointDataProvider**: Returns a single value per call. Best for
  database-backed lookups, CGAN generators, or memory-constrained scenarios.

Both protocols use structural typing (``typing.Protocol``), so consumers
do NOT need to import or explicitly subclass them — any object with the
right method signature will work.

Example — batch provider from a CSV::

    class CsvProvider:
        def __init__(self, path: str):
            self._df = pd.read_csv(path, parse_dates=["timestamp"])

        def get_fundamental_series(self, symbol, start, end):
            s = self._df.set_index("timestamp")[symbol]
            idx = pd.to_datetime([start, end], unit="ns")
            return s.loc[idx[0]:idx[1]].astype(int)

Example — point provider from a database::

    class DbProvider:
        def __init__(self, conn):
            self._conn = conn

        def get_fundamental_at(self, symbol, timestamp):
            row = self._conn.execute(
                "SELECT price FROM fundamentals WHERE symbol=? AND ts<=? ORDER BY ts DESC LIMIT 1",
                (symbol, timestamp),
            ).fetchone()
            return int(row[0])
"""

from enum import Enum
from typing import Protocol, runtime_checkable

import pandas as pd
from abides_core import NanosecondTime

# ---------------------------------------------------------------------------
# Interpolation strategy for batch mode
# ---------------------------------------------------------------------------


class InterpolationStrategy(Enum):
    """How to resolve timestamp mismatches in batch mode.

    When an agent queries a timestamp that does not exactly match a data point
    in the provider's series, the oracle uses this strategy to determine the
    returned value.
    """

    FORWARD_FILL = "ffill"
    """Use the last known value (realistic for market data)."""

    NEAREST = "nearest"
    """Use the value at the closest timestamp."""

    LINEAR = "linear"
    """Linearly interpolate between the two surrounding data points."""


# ---------------------------------------------------------------------------
# Provider protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class BatchDataProvider(Protocol):
    """Provider that returns an entire fundamental value series at once.

    Best for: file-based data, pre-computed series, small-to-medium datasets
    that fit comfortably in memory.

    The returned ``pd.Series`` must:
    - Have a ``DatetimeIndex`` (timezone-naive, nanosecond precision)
    - Contain integer values representing prices in cents
    - Be sorted by index in ascending order
    """

    def get_fundamental_series(
        self, symbol: str, start: NanosecondTime, end: NanosecondTime
    ) -> pd.Series: ...


@runtime_checkable
class PointDataProvider(Protocol):
    """Provider that returns a single fundamental value per call.

    Best for: database queries, CGAN/generative model inference, very large
    datasets, or memory-constrained environments.

    Returns an integer price in cents for the given symbol and timestamp.
    """

    def get_fundamental_at(self, symbol: str, timestamp: NanosecondTime) -> int: ...


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


class DataFrameProvider:
    """Concrete ``BatchDataProvider`` backed by a dict of pd.Series.

    This is a convenience class for testing and simple file-based workflows.
    It wraps pre-loaded data and slices it according to the requested range.

    Args:
        data: Mapping of symbol name to a ``pd.Series`` with a
              ``DatetimeIndex`` and integer values (cents).
    """

    def __init__(self, data: dict[str, pd.Series]) -> None:
        self._data = data

    def get_fundamental_series(
        self, symbol: str, start: NanosecondTime, end: NanosecondTime
    ) -> pd.Series:
        """Return the slice of data for *symbol* between *start* and *end*.

        Raises:
            KeyError: If *symbol* is not in the provider's data.
        """
        series = self._data[symbol]
        start_ts = pd.Timestamp(start, unit="ns")
        end_ts = pd.Timestamp(end, unit="ns")
        return series.loc[start_ts:end_ts]

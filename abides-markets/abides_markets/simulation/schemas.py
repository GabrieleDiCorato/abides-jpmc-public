"""Pandera DataFrameModel schemas for all typed DataFrames in the simulation package.

Design notes
------------
* ``L1DataFrameSchema`` and ``L2DataFrameSchema`` use ``strict=True`` — their
  shapes are fully known and no extra columns should pass through.
* ``RawLogsSchema`` and ``OrderLogsSchema`` use ``strict="filter"`` — the raw
  log DataFrame is heterogeneous by design (different event types widen it with
  different extra columns).  The schema only guarantees its declared columns
  are present, correctly typed, and satisfy their constraints; additional
  columns are silently retained.
* All price / quantity columns use ``gt=0`` (or ``ge=0`` where zero is
  semantically valid, e.g. timestamps).  In particular ``L2DataFrameSchema``
  enforces ``price_cents: gt=0`` and ``qty: gt=0`` — this makes it structurally
  impossible for zero-padded book levels to survive pandera validation.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import DataFrameModel
from pandera.typing.pandas import Series


class L1DataFrameSchema(pa.DataFrameModel):
    """Schema for the wide-format L1 bid/ask snapshot DataFrame.

    Produced by :meth:`~abides_markets.simulation.L1Snapshots.as_dataframe`.

    All price/quantity fields are nullable because either side of the book may
    be empty (e.g. at market open or after a one-sided sweep).
    """

    time_ns: Series[pa.Int64] = pa.Field(ge=0, description="Event timestamp (ns, Unix epoch).")
    bid_price_cents: Series[pa.Int64] = pa.Field(
        ge=0, nullable=True, description="Best bid price in integer cents; None if no bid."
    )
    bid_qty: Series[pa.Int64] = pa.Field(
        ge=0, nullable=True, description="Quantity resting at best bid; None if no bid."
    )
    ask_price_cents: Series[pa.Int64] = pa.Field(
        ge=0, nullable=True, description="Best ask price in integer cents; None if no ask."
    )
    ask_qty: Series[pa.Int64] = pa.Field(
        ge=0, nullable=True, description="Quantity resting at best ask; None if no ask."
    )

    class Config:
        strict = True
        coerce = False


class L2DataFrameSchema(pa.DataFrameModel):
    """Schema for the long/tidy sparse L2 order-book snapshot DataFrame.

    Produced by :meth:`~abides_markets.simulation.L2Snapshots.as_dataframe`.

    Key guarantees:
    * ``price_cents > 0`` and ``qty > 0`` — zero-padded phantom levels are
      architecturally banned at the source and enforced here at the schema
      boundary.
    * ``level`` is 0-indexed: 0 is the best price on each side.
    * Each ``(time_ns, side)`` group contains only *observed* resting levels —
      the number of rows per group equals the instantaneous book depth, which
      is itself a market microstructure signal.
    """

    time_ns: Series[pa.Int64] = pa.Field(ge=0, description="Event timestamp (ns, Unix epoch).")
    side: Series[str] = pa.Field(
        isin=["bid", "ask"], description="Book side: 'bid' or 'ask'."
    )
    level: Series[pa.Int64] = pa.Field(
        ge=0, description="Depth level, 0-indexed (0 = best price)."
    )
    price_cents: Series[pa.Int64] = pa.Field(
        gt=0, description="Limit price in integer cents. Never zero (sparse representation)."
    )
    qty: Series[pa.Int64] = pa.Field(
        gt=0, description="Aggregate resting quantity. Never zero (sparse representation)."
    )

    class Config:
        strict = True
        coerce = False


class RawLogsSchema(pa.DataFrameModel):
    """Base schema for the full ``parse_logs_df()`` output.

    Only the four columns that are **always** present across all event types
    are declared here.  All additional event-specific columns (e.g.
    ``symbol``, ``order_id``) pass through untouched thanks to
    ``strict="filter"``.
    """

    EventTime: Series[pa.Int64] = pa.Field(ge=0, description="Event timestamp (ns, Unix epoch).")
    EventType: Series[str] = pa.Field(description="Event label, e.g. 'ORDER_SUBMITTED'.")
    agent_id: Series[pa.Int64] = pa.Field(description="ID of the agent that logged the event.")
    agent_type: Series[str] = pa.Field(description="Type string of the agent.")

    class Config:
        strict = False
        coerce = False


_ORDER_EVENT_TYPES = frozenset(
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


class OrderLogsSchema(RawLogsSchema):
    """Schema for the order-event subset of the log DataFrame.

    Produced by :meth:`~abides_markets.simulation.SimulationResult.order_logs`.

    Inherits the four base columns from :class:`RawLogsSchema` and adds the
    columns that are guaranteed on every order-related log entry.  Extra
    per-order-type columns (``limit_price``, ``is_hidden``, etc.) remain
    accessible but are not declared here.
    """

    symbol: Series[str] = pa.Field(description="Trading symbol.")
    order_id: Series[pa.Int64] = pa.Field(description="Unique order identifier.")
    quantity: Series[pa.Int64] = pa.Field(gt=0, description="Order quantity in shares.")
    side: Series[str] = pa.Field(
        isin=["BID", "ASK"], description="Order side: 'BID' or 'ASK'."
    )
    fill_price: Series[pa.Int64] = pa.Field(
        gt=0,
        nullable=True,
        description="Fill price in cents; None unless ORDER_EXECUTED.",
    )
    limit_price: Series[pa.Int64] = pa.Field(
        gt=0,
        nullable=True,
        description="Limit price in cents; None for market orders.",
    )

    class Config:
        strict = False
        coerce = False

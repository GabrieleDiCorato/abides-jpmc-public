from abc import ABC
from dataclasses import dataclass
from typing import Any

from abides_core import Message


@dataclass
class QueryMsg(Message, ABC):
    symbol: str


@dataclass
class QueryResponseMsg(Message, ABC):
    symbol: str
    mkt_closed: bool


@dataclass
class QueryLastTradeMsg(QueryMsg):
    # Inherited Fields:
    # symbol: str
    pass


@dataclass
class QueryLastTradeResponseMsg(QueryResponseMsg):
    # Inherited Fields:
    # symbol: str
    # mkt_closed: bool
    last_trade: int | None


@dataclass
class QuerySpreadMsg(QueryMsg):
    # Inherited Fields:
    # symbol: str
    depth: int


@dataclass
class QuerySpreadResponseMsg(QueryResponseMsg):
    # Inherited Fields:
    # symbol: str
    # mkt_closed: bool
    depth: int
    bids: list[tuple[int, int]]
    asks: list[tuple[int, int]]
    last_trade: int | None


@dataclass
class QueryOrderStreamMsg(QueryMsg):
    # Inherited Fields:
    # symbol: str
    length: int


@dataclass
class QueryOrderStreamResponseMsg(QueryResponseMsg):
    # Inherited Fields:
    # symbol: str
    # mkt_closed: bool
    length: int
    orders: list[dict[str, Any]]


@dataclass
class QueryTransactedVolMsg(QueryMsg):
    # Inherited Fields:
    # symbol: str
    lookback_period: str


@dataclass
class QueryTransactedVolResponseMsg(QueryResponseMsg):
    # Inherited Fields:
    # symbol: str
    # mkt_closed: bool
    bid_volume: int
    ask_volume: int

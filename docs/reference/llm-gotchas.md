# ABIDES-NG - LLM Integration Reference & Gotchas

Canonical reference for **AI coding assistants** building on ABIDES-NG.
Covers the async agent model, every source of `None`/`NaN`/empty collections,
safe-access patterns, and the simulation runner API.

---

## Table of Contents

1. [Mental Model — Not a Loop, Not Synchronous](#1-mental-model)
2. [Agent Lifecycle](#2-agent-lifecycle)
3. [Market Data — All Sources of None / Empty](#3-market-data--all-sources-of-none--empty)
4. [Pre-Market Window — Special State](#4-pre-market-window--special-state)
5. [Order Book Data Structures Explained](#5-order-book-data-structures-explained)
6. [The mark_to_market Trap](#6-the-mark_to_market-trap)
7. [Price Units — Everything is Integer Cents](#7-price-units--everything-is-integer-cents)
8. [Order Lifecycle and Tracking](#8-order-lifecycle-and-tracking)
9. [Correct Safe-Access Patterns](#9-correct-safe-access-patterns)
10. [Subscription vs. Pull-Based Data](#10-subscription-vs-pull-based-data)
11. [Complete State Validity Checklist](#11-complete-state-validity-checklist)
12. [Running Simulations](#12-running-simulations)
13. [External Oracle — Historical / Generated Data](#13-external-oracle--historical--generated-data)
14. [logEvent Deep Copy](#14-logevent-deep-copy)

---

## 1. Mental Model

ABIDES is a **discrete-event simulation** — NOT a loop-based system.

```
❌ WRONG mental model:
    while market_open:
        bid, ask = get_spread()   # synchronous
        if bid is not None:
            place_order(...)

✅ CORRECT mental model:
    def wakeup(self, current_time):
        # 1. Request data (sends a message - async)
        self.get_current_spread(symbol)
        # 2. Return — execution ends here

    def receive_message(self, current_time, sender_id, message):
        # 3. Data arrives HERE, possibly nanoseconds later
        if isinstance(message, QuerySpreadResponseMsg):
            # NOW self.known_bids / known_asks are fresh
            place_order(...)
```

**Key consequence**: Every call to `get_current_spread()`, `get_last_trade()`, etc.
sends a *message* to the Exchange. Nothing is returned immediately. The response
arrives asynchronously in `receive_message()`.

---

## 2. Agent Lifecycle

Custom agents subclass `TradingAgent` and implement exactly two callbacks:

| Callback | Called when | Typical actions |
|---|---|---|
| `wakeup(current_time)` | At each scheduled time | `set_wakeup()` for next tick; request data; place orders |
| `receive_message(current_time, sender_id, message)` | When a message arrives | React to fills, spread responses, market data |

`TradingAgent` handles all other lifecycle phases internally (`kernel_starting`,
`kernel_stopping`, mark-to-market at close, etc.). **Do not override these.**

```python
class MyAgent(TradingAgent):
    def wakeup(self, current_time):
        can_trade = super().wakeup(current_time)  # ALWAYS call super first
        if not can_trade:
            return
        self.get_current_spread(self.symbol)      # send data request
        self.set_wakeup(current_time + self.wake_freq)  # schedule next tick

    def receive_message(self, current_time, sender_id, message):
        super().receive_message(current_time, sender_id, message)  # ALWAYS call super
        if isinstance(message, QuerySpreadResponseMsg):
            self._on_spread(message)
```

> **Subscribe-mode agents**: call `super().wakeup()` but proceed with
> one-time subscription setup even when it returns `False` — subscriptions
> must be registered before market hours arrive.

---

## 3. Market Data — All Sources of None / Empty

### 3.1 `self.known_bids[symbol]` and `self.known_asks[symbol]`

| Situation | Value |
|-----------|-------|
| Before any spread query response received | **`KeyError`** — key doesn't exist |
| After response, but book side is empty | **`[]`** (empty list) |
| After response, book side has data | `[(price, qty), ...]` |

```python
# ❌ Crashes before first response
bid = self.known_bids["ABM"][0][0]

# ❌ Crashes if book side is empty
bid = self.known_bids["ABM"][0][0]  # IndexError on []

# ✅ Safe access
bids = self.known_bids.get("ABM", [])
bid = bids[0][0] if bids else None
```

### 3.2 `self.last_trade[symbol]`

| Situation | Value |
|-----------|-------|
| Before any trade has occurred | **`KeyError`** — key doesn't exist |
| After any trade in the session | `int` (price in cents) |
| After market close message | Updated to official close price |

```python
# ❌ Crashes before first trade (common at simulation start)
price = self.last_trade["ABM"]

# ✅ Safe access
price = self.last_trade.get("ABM")  # Returns None if missing
```

### 3.3 `self.mkt_open` and `self.mkt_close`

Both start as `None`. They are only populated after the agent receives a
`MarketHoursMsg` from the Exchange (which arrives in response to the automatic
`MarketHoursRequestMsg` sent within `TradingAgent.wakeup()`).

```python
# ❌ Crashes on first wakeup before market hours are known
next_wakeup = self.mkt_open + interval

# ✅ TradingAgent.wakeup() returns a boolean — use it
def wakeup(self, current_time):
    can_trade = super().wakeup(current_time)
    if not can_trade:
        return  # mkt_open/mkt_close not yet known, or market closed
    ...
```

### 3.4 `L1DataMsg.bid` and `L1DataMsg.ask`

The type annotation says `Tuple[int, int]` but in practice these come from
`OrderBook.get_l1_bid_data()` / `get_l1_ask_data()` which return **`None`**
when the book side is empty.

```python
# The actual source in order_book.py:
def get_l1_bid_data(self) -> Optional[Tuple[int, int]]:
    if len(self.bids) == 0:
        return None   # ← this happens before any orders are placed

# ✅ Always guard
if message.bid is not None:
    bid_price, bid_qty = message.bid
```

### 3.5 `L2DataMsg.bids` and `L2DataMsg.asks`

These are always `List[Tuple[int, int]]` — never `None`. But they can be
`[]` (empty list) when the book side has no resting orders.

```python
bids = message.bids  # List — safe, but may be []
if bids:
    best_bid_price = bids[0][0]  # ✅ guarded
    best_bid_qty   = bids[0][1]
```

### 3.6 `QuerySpreadResponseMsg.last_trade`

Type annotation: `Optional[int]`. Value is `None` when no trade has happened
in the session yet.

```python
# ❌ Crashes before first trade
spread_msg.last_trade + 100

# ✅ Safe
if spread_msg.last_trade is not None:
    ...
```

---

## 4. Pre-Market Window — Special State

The **pre-market window** is the period between simulation start and `mkt_open`.
LLM-generated agents frequently crash during this window because:

1. `wakeup()` is called, but `mkt_open` is `None`
2. `known_bids`/`known_asks` dictionaries are empty
3. `last_trade` dictionary is empty
4. The order book itself has no orders yet → all queries return `None` / `[]`

The **correct pattern** (already implemented in `TradingAgent`):

```python
def wakeup(self, current_time):
    can_trade = super().wakeup(current_time)

    # super().wakeup() returns False if:
    #   - mkt_open is None (market hours not yet known)
    #   - mkt_close is None
    #   - self.mkt_closed is True
    if not can_trade:
        return   # ← Do nothing during pre-market

    # Only here is it safe to trade
```

---

## 5. Order Book Data Structures Explained

### L1 — Best bid/ask only

```python
# OrderBook.get_l1_bid_data() → Optional[Tuple[int, int]]
# Returns: (price_in_cents, quantity)  OR  None if book side empty

(100_050, 200)   # best bid: $1000.50 for 200 shares
None             # no bids at all
```

### L2 — All price levels, aggregated by price

```python
# OrderBook.get_l2_bid_data(depth=N) → List[Tuple[int, int]]
# Each tuple: (price_in_cents, total_quantity_at_that_price)
# Ordered from best price inward (closest to spread first)

[(100_100, 500), (100_050, 200), (100_000, 1000)]
# Index 0 = best bid, index -1 = worst bid

[]   # Empty list if no bids at all
```

### L3 — All price levels, per-order quantities

```python
# OrderBook.get_l3_bid_data(depth=N) → List[Tuple[int, List[int]]]
# Each tuple: (price_in_cents, [qty_order1, qty_order2, ...])
# Within a price level, orders are in FIFO priority order

[(100_100, [200, 300]), (100_050, [500])]
```

### TradingAgent cache fields

| Field | Type | Content |
|-------|------|---------|
| `self.known_bids[symbol]` | `List[Tuple[int,int]]` | L2 bid levels, best first. May be `[]`. **KeyError if symbol never queried.** |
| `self.known_asks[symbol]` | `List[Tuple[int,int]]` | L2 ask levels, best first. May be `[]`. **KeyError if symbol never queried.** |
| `self.last_trade[symbol]` | `int` | Last executed price (cents). **KeyError if no trade yet.** |
| `self.exchange_ts[symbol]` | `NanosecondTime` | Exchange timestamp of last L2 update. Populated by subscriptions only. |
| `self.mkt_open` | `Optional[int]` | Nanoseconds. `None` until `MarketHoursMsg` received. |
| `self.mkt_close` | `Optional[int]` | Nanoseconds. `None` until `MarketHoursMsg` received. |

---

## 6. The `mark_to_market` Trap

`TradingAgent.mark_to_market(holdings)` does:

```python
value = self.last_trade[symbol] * shares  # ← KeyError if no trades!
```

**This will crash if called before any trade has occurred in the session.**
The `kernel_stopping()` method calls `mark_to_market()` — so even if your
agent never trades, this runs at the end of every simulation.

The Exchange sends `MarketClosePriceMsg` at market close with all final
prices, which populates `last_trade`. So crashes are only a risk if called
*before* the market has had any trades.

```python
# ✅ Safe pattern
def compute_portfolio_value(self):
    if self.last_trade.get(self.symbol) is None:
        return None  # Not enough data yet
    return self.mark_to_market(self.holdings)
```

---

## 7. Price Units — Everything is Integer Cents

**All prices in ABIDES are integers in cents.** There are no floats for prices.

```python
# $100.00 = 10_000 cents
# $1.50   =    150 cents
# $0.01   =      1 cent

# ❌ Common LLM mistake: treating prices as dollars
if bid_price > 100:      # Wrong — this is 100 cents = $1.00
    ...

# ✅ Correct
if bid_price > 10_000:   # $100.00
    ...

# Converting for display only:
print(f"Bid: ${bid_price / 100:.2f}")
```

**Consequence for midpoint calculation:**

```python
# ✅ Integer midpoint (standard in ABIDES)
midpoint = (bid + ask) // 2

# ✅ Also used in TradingAgent.get_known_bid_ask_midpoint():
midpoint = int(round((bid + ask) / 2))
```

---

## 8. Order Lifecycle and Tracking

### Agent's view of its own orders

```python
self.orders: Dict[int, Order]
# Keys: order_id
# Present: order is open (submitted, not fully filled, not cancelled)
# Absent: order fully filled OR cancelled

# ⚠️ Order may disappear from self.orders BEFORE you receive the
#    OrderExecutedMsg — timing race in the event queue
```

### Order execution flow

```
place_limit_order() → LimitOrderMsg → Exchange
                                         ↓
                               (waits for counterpart)
                                         ↓
                             OrderAcceptedMsg ← agent.order_accepted()
                                         ↓
                             OrderExecutedMsg ← agent.order_executed()
                             (holdings & CASH updated automatically by TradingAgent)
```

### Cancellation subtlety

```python
# Calling cancel_order() does NOT immediately remove from self.orders.
# The order stays in self.orders until OrderCancelledMsg arrives.
# If the order executes BEFORE the cancel reaches the Exchange,
# you receive OrderExecutedMsg instead — no cancel confirmation.
```

---

## 9. Correct Safe-Access Patterns

### Getting bid/ask safely

```python
def _safe_best_bid_ask(self, symbol: str):
    """Returns (bid, ask) in cents, or (None, None) if unavailable."""
    bids = self.known_bids.get(symbol, [])
    asks = self.known_asks.get(symbol, [])
    bid = bids[0][0] if bids else None
    ask = asks[0][0] if asks else None
    return bid, ask
```

### Getting midpoint safely

```python
def _safe_midpoint(self, symbol: str) -> Optional[int]:
    bid, ask = self._safe_best_bid_ask(symbol)
    if bid is None or ask is None:
        return None
    return (bid + ask) // 2
```

### Checking if ready to trade

```python
def _can_trade(self, symbol: str) -> bool:
    """True only if all the data we need is present."""
    if self.mkt_open is None or self.mkt_close is None:
        return False
    if self.mkt_closed:
        return False
    bids = self.known_bids.get(symbol, [])
    asks = self.known_asks.get(symbol, [])
    return bool(bids and asks)
```

### Computing spread

```python
def _spread(self, symbol: str) -> Optional[int]:
    bid, ask = self._safe_best_bid_ask(symbol)
    if bid is None or ask is None:
        return None
    return ask - bid   # in cents; should always be >= 0 in a valid book
```

---

## 10. Subscription vs. Pull-Based Data

ABIDES supports two ways to get market data:

### Pull (request/response)

```python
# In wakeup():
self.get_current_spread(symbol)          # → QuerySpreadResponseMsg
self.get_last_trade(symbol)              # → QueryLastTradeResponseMsg
self.get_transacted_volume(symbol)       # → QueryTransactedVolResponseMsg

# Each sends a message; response arrives in receive_message() LATER.
```

### Subscription (push)

```python
# In wakeup(), first time only:
from abides_markets.messages.marketdata import L2SubReqMsg
self.request_data_subscription(L2SubReqMsg(
    symbol=symbol,
    freq=int(1e8),   # push every 100ms of sim time
    depth=10,        # top 10 levels
))

# Updates arrive as L2DataMsg in receive_message() periodically.
# TradingAgent.handle_market_data() automatically updates:
#   self.known_bids[symbol]
#   self.known_asks[symbol]
#   self.last_trade[symbol]
#   self.exchange_ts[symbol]
```

**Recommendation**: Use **subscriptions** if your strategy needs fresh data on
every tick. Use **pull** if you only need data occasionally.

**Available subscription types:**

| Message Class | Data | Trigger |
|---------------|------|---------|
| `L1SubReqMsg` | Best bid/ask (price + qty) | Periodic |
| `L2SubReqMsg` | All price levels (aggregated) | Periodic |
| `L3SubReqMsg` | All price levels (per-order) | Periodic |
| `TransactedVolSubReqMsg` | Buy/sell volume in lookback window | Periodic |
| `BookImbalanceSubReqMsg` | Imbalance event | Event-driven |

---

## 11. Complete State Validity Checklist

Use this checklist before acting on any market state in your agent:

```python
def _state_is_valid(self, symbol: str) -> bool:
    # 1. Do we know market hours?
    if self.mkt_open is None or self.mkt_close is None:
        return False

    # 2. Is the market still open?
    if self.mkt_closed:
        return False

    # 3. Have we received at least one book update?
    if symbol not in self.known_bids or symbol not in self.known_asks:
        return False

    # 4. Does the book have both sides?
    if not self.known_bids[symbol] or not self.known_asks[symbol]:
        return False

    # 5. Is the spread valid (no crossed book)?
    bid = self.known_bids[symbol][0][0]
    ask = self.known_asks[symbol][0][0]
    if bid >= ask:
        return False  # Crossed book — shouldn't happen in well-formed sim

    return True
```

---

## Quick Reference: What Returns None/Empty and When

| Expression | Type | `None`/empty when |
|------------|------|-------------------|
| `self.mkt_open` | `Optional[int]` | Before `MarketHoursMsg` received |
| `self.mkt_close` | `Optional[int]` | Before `MarketHoursMsg` received |
| `self.known_bids.get(symbol)` | `Optional[List]` | Before first spread response (`None`); or empty book (`[]`) |
| `self.known_asks.get(symbol)` | `Optional[List]` | Before first spread response (`None`); or empty book (`[]`) |
| `self.last_trade.get(symbol)` | `Optional[int]` | Before any trade in the session |
| `L1DataMsg.bid` | `Optional[Tuple]` | When bid side of book is empty |
| `L1DataMsg.ask` | `Optional[Tuple]` | When ask side of book is empty |
| `L2DataMsg.bids` | `List` | Never `None`, but can be `[]` |
| `L2DataMsg.asks` | `List` | Never `None`, but can be `[]` |
| `QuerySpreadResponseMsg.last_trade` | `Optional[int]` | Before any trade |
| `QueryLastTradeResponseMsg.last_trade` | `Optional[int]` | Before any trade |
| `OrderBook.get_l1_bid_data()` | `Optional[Tuple]` | Empty book side |
| `OrderBook.get_l1_ask_data()` | `Optional[Tuple]` | Empty book side |
| `OrderBook.get_l2_bid_data()` | `List` | Never `None`, but can be `[]` |
| `OrderBook.get_l2_ask_data()` | `List` | Never `None`, but can be `[]` |
| `OrderBook.last_trade` | `Optional[int]` | Before any trade |

---

## 12. Running Simulations

```python
from abides_markets.config_system import SimulationBuilder
from abides_markets.simulation import run_simulation, ResultProfile

config = (SimulationBuilder()
    .from_template("rmsc04")
    .enable_agent("my_strategy", count=1, threshold=0.08)
    .seed(42)
    .build())

result = run_simulation(config, profile=ResultProfile.QUANT)
```

**`ResultProfile` tiers:**

| Profile | Includes | Use when |
|---|---|---|
| `SUMMARY` (default) | PnL, liquidity, L1 close | REST responses, quick checks |
| `QUANT` | + L1/L2 series, trade attribution, equity curves | Backtesting, analytics |
| `FULL` | + raw agent log DataFrame | Debugging agent behaviour |

**Key `SimulationResult` fields:**

| Field | Type | Content |
|---|---|---|
| `result.metadata` | `SimulationMetadata` | Seed, tickers, timing |
| `result.agents` | `list[AgentData]` | Per-agent PnL, holdings |
| `result.markets` | `dict[str, MarketSummary]` | L1 close, liquidity, optional L1/L2 series |
| `result.logs` | `DataFrame \| None` | Raw agent events (`FULL` only) |
| `result.extensions` | `dict[str, Any]` | Custom extractor outputs |

**Parallel runs:**

```python
from abides_markets.simulation import run_batch

results = run_batch([cfg1, cfg2, cfg3], n_workers=4, profile=ResultProfile.QUANT)
```

> Full API: `config-system.md`, `metrics-api.md`, `parallel-simulation.md`

---

## 13. External Oracle — Historical / Generated Data

> **Important:** `Oracle` is an `abc.ABC`. Any custom oracle must implement both
> `get_daily_open_price(...)` and `observe_price(...)`.

```python
from abides_markets.oracles import ExternalDataOracle

oracle = ExternalDataOracle(
    mkt_open, mkt_close, ["AAPL"],
    data={"AAPL": my_series},  # pd.Series with DatetimeIndex, values in integer cents
)

config = (SimulationBuilder()
    .oracle_instance(oracle)   # inject pre-built oracle
    .enable_agent("noise", count=500)
    .seed(42)
    .build())
```

Alternatively, declare `ExternalDataOracleConfig` in the config (marker type) and pass
the oracle at runtime:

```python
result = run_simulation(config, oracle_instance=my_oracle)
```

> Full pattern, `DataFrameProvider`, and oracle-less simulations:
> `custom-agent-guide.md §7`

---

## 14. logEvent Deep Copy

`logEvent(event_type, event, deepcopy_event=False)` — default is **no deep copy**.
Logging a mutable object and then mutating it means the log entry reflects
the *final* state, not the state at log time.

```python
# BAD — dict is mutated after logging
self.logEvent("SNAPSHOT", self.holdings)

# GOOD — snapshot is frozen at log time
self.logEvent("SNAPSHOT", self.holdings, deepcopy_event=True)

# OK — immutable primitives don't need deepcopy
self.logEvent("TRADE", {"price": price, "qty": qty})
```

---

## Common Crashes — Quick Reference

| Crash / symptom | Root cause | Fix |
|---|---|---|
| `KeyError: 'ABM'` on `self.known_bids` | No spread response received yet | `.get(symbol, [])` |
| `IndexError` on `self.known_bids[symbol][0]` | Book side is `[]` | `if bids:` guard |
| `KeyError: 'ABM'` on `self.last_trade` | No trade executed yet | `.get(symbol)` |
| `TypeError: unsupported operand None + int` | `mkt_open` is `None` | Check `super().wakeup()` return |
| Prices wrong by factor of 100 | Treating cents as dollars | All prices are integer cents |
| `RuntimeError` at `kernel_stopping` | `mark_to_market` called before first trade | Guard with `self.last_trade.get(symbol)` |
| `ValueError: Kernel agents list violates agents[i].id == i` | Agent injected with non-matching id | Use `config_add_agents()` (auto-reassigns) or set `agent.id = index` before passing to `Kernel(...)` |

---

## Kernel invariants (added 2026-04)

The kernel maintains a strict invariant: **for every agent in
`Kernel.__init__(agents=...)`, `agents[i].id == i`**. The kernel
indexes its parallel per-agent state arrays by agent id, and a
mismatch would silently corrupt routing.

If you build the agents list manually, assign ids by enumeration:

```python
agents = [...]
for i, a in enumerate(agents):
    a.id = i
kernel = Kernel(agents=agents, ...)
```

The `config_add_agents()` helper in `abides_markets.utils` does this
reassignment automatically when appending runtime agents to a built
config.

## `Kernel.initialize()` is the canonical reset point (added 2026-04)

`Kernel.initialize()` clears per-run state before notifying agents, so
it is safe to call repeatedly (gym training loops, parallel workers in
the same interpreter, custom multi-episode harnesses). The following
state is reset to a clean slate every call:

- `messages` (event heap)
- `custom_state` (the dict returned by `terminate()`)
- `summary_log`
- `_run_stats` (per-run scheduling stats: wall-clock anchor + message counter)
- `current_agent_additional_delay`
- `agent_current_times` (reset to `start_time`)

`agent_computation_delays` is **not** cleared — it carries per-agent
overrides from the constructor that must persist across resets. If an
agent mutates its own delay at runtime via `set_agent_compute_delay()`,
that mutation also survives across `initialize()`.

`Kernel.reset()` calls `terminate()` then `initialize()`; the actual
clearing lives in `initialize()` so any direct call to it gets the
same clean slate.

## Reserved `custom_properties` keys (added 2026-04)

`Kernel(custom_properties={...})` rejects keys that would shadow
kernel-managed attributes (`agents`, `messages`, `random_state`,
`current_time`, latency fields, etc.). Use this dict only for
user-defined extras like `oracle`. The full blocklist is in
`abides_core.kernel._KERNEL_RESERVED_ATTRS`.

## Always seed the kernel (added 2026-04)

Constructing a `Kernel` without an explicit `seed=` or `random_state=`
emits `DeprecationWarning` and falls back to OS entropy. Notebooks and
ad-hoc scripts should pass `seed=<int>` or
`random_state=np.random.RandomState(<int>)` for reproducibility.

## Heap entries are `_HeapEntry` dataclasses (added 2026-04)

The kernel's event queue stores `_HeapEntry(deliver_at, seq,
sender_id, recipient_id, message)` instances, ordered by `(deliver_at,
seq)`. `seq` is a per-kernel monotonic counter reset by
`Kernel.initialize()`. Do not push raw tuples onto `kernel.messages`
— use `Kernel._enqueue(...)` (or the public `send_message` /
`set_wakeup` APIs).

`Message.__lt__` no longer exists; comparing two `Message` instances
will raise `TypeError`. `Message.message_id` survives only as a
deprecated property returning `id(self)` and emits
`DeprecationWarning` when accessed.

`Kernel.set_wakeup` reuses a module-level `_WAKEUP_SINGLETON`. Do not
mutate `WakeupMsg` instances — they may be shared.

## `Agent.delay()` works inside `receive_message()` (fixed 2026-04)

Prior to PR 3 of the kernel improvements, calls to `self.delay(N)`
made inside `receive_message()` were silently dropped because the
kernel advanced the agent's "busy until" time *before* dispatch. The
runner now advances `agent_current_times[recipient_id]` *after*
dispatch, so `delay()` calls inside both `wakeup()` and
`receive_message()` correctly shift the agent's next slot.

## Per-agent-type metrics live in `custom_state` (added 2026-04)

The kernel no longer carries finance-specific
`mean_result_by_agent_type` / `agent_count_by_type` defaultdicts.
Use `Agent.report_metric(key, value)` to aggregate any numeric metric
by agent type. Storage layout:

```python
kernel.custom_state["agent_type_metrics"][agent_type][key] = {
    "sum": float,
    "count": int,
}
```

`Kernel.terminate()` prints `mean = sum/count` for every reported
`(type, key)` pair with `count > 0`. `TradingAgent.kernel_stopping()`
already reports its mark-to-market gain as `"ending_value"`.

External code that read the old kernel attributes must switch to
reading `custom_state["agent_type_metrics"]`. External code that
wrote them must switch to `Agent.report_metric()`.

## `LatencyModel.get_latency` requires `random_state` kwarg (added 2026-04)

`Kernel.send_message` always routes through `agent_latency_model`
now, even when the caller did not pass one (the kernel wraps the
legacy `agent_latency` / `default_latency` / `latency_noise` args
into a `MatrixLatencyModel` or `UniformLatencyModel`).

The kernel passes `random_state=self.random_state` to every
`get_latency` call. Custom `LatencyModel` subclasses **must** accept
`random_state` as a keyword-only parameter:

```python
def get_latency(
    self,
    sender_id: int,
    recipient_id: int,
    *,
    random_state: np.random.RandomState | None = None,
) -> int:
    ...
```

The built-in cubic `LatencyModel` accepts the kwarg but ignores it,
keeping bit-for-bit reproducibility for cubic users.

`Kernel.agent_latency` and `Kernel.latency_noise` no longer exist —
read the underlying data from the model object directly.

## Latency models default to no RNG draw (changed 2026-04, BREAKING)

`UniformLatencyModel` and `MatrixLatencyModel` default to
`noise=None`. When `noise` is `None`, `get_latency` returns its
configured latency directly and does **not** touch the kernel RNG.

This is a deliberate reproducibility break versus pre-PR-5b runs,
which always consumed one `np.random.choice([1.0])` draw per
`send_message` even when no noise was configured. To restore the
legacy bit-for-bit behavior, pass `noise=[1.0]` explicitly:

```python
model = UniformLatencyModel(latency=1_000_000_000, noise=[1.0])
```

The `Kernel` constructor's legacy `latency_noise` kwarg still
forwards into the wrapped model unchanged, so callers that already
pass `latency_noise=[1.0]` are unaffected.

## Per-agent state is numpy `int64` (changed 2026-04)

`Kernel.agent_current_times` and `Kernel.agent_computation_delays`
are now `numpy.ndarray[int64]`, exposed internally as
`Kernel._agent_current_times` and `Kernel._agent_computation_delays`.
The legacy public names remain as **read-only deprecation
properties** that emit a one-shot `DeprecationWarning` and return
a non-writable `ndarray` view.

Implications:

- External *readers* keep working (with one warning per attribute
  per process) but should migrate to the underscore names if they
  are inside `abides_core`.
- External *writers* will fail loudly: the returned view rejects
  item assignment. Use `kernel.set_agent_compute_delay(agent_id, ns)`
  for runtime delay updates, or pass `agent_computation_delays` (an
  `np.ndarray` with `dtype=int64`, shape `(n_agents,)`) to the
  `Kernel` constructor for static overrides.
- `numpy.int64` arithmetic does not auto-promote to Python `int`
  inside f-strings and JSON serialisers. Cast at the boundary if
  you read these arrays directly:

  ```python
  finish_time = int(kernel._agent_current_times[agent_id])
  ```

  The kernel itself already casts
  `custom_state["kernel_slowest_agent_finish_time"]` to a Python
  `int` so notebooks and log parsers stay clean.

`Kernel.find_agents_by_type` is now O(1): the kernel pre-indexes
every agent under each class in its MRO at construction time, so
passing a base class still returns every subclass instance
(`isinstance` semantics are preserved).

## Kernel constructor is keyword-only (changed 2026-04)

Every parameter on `Kernel.__init__` after `agents` is keyword-only.
Calling `Kernel(my_agents, str_to_ns("09:30:00"), …)` raises
`TypeError`. Use `Kernel(agents=my_agents, start_time=…, …)` instead.

## Kernel lifecycle is enforced (added 2026-04)

`Kernel` tracks its own state through four positions
(`CONSTRUCTED → INITIALIZED → RUNNING → TERMINATED`) and validates
transitions at the public method boundary:

| Method | Allowed inbound state |
|--------|----------------------|
| `initialize()` | `CONSTRUCTED`, `TERMINATED` (post-`reset` re-init) |
| `runner()` | `INITIALIZED`, `RUNNING` (gym yields keep state at `RUNNING`) |
| `terminate()` | `INITIALIZED`, `RUNNING` |
| `reset()` | any state — terminates first if needed, then initialises |

Out-of-order calls raise `RuntimeError` with the offending state and
the list of allowed states. The legacy public attribute `has_run`
remains as a bool for back-compat.

## Inject a `GymAdapter`; auto-detection is deprecated (added 2026-04)

If you build a `Kernel` with an `abides-gym` experimental agent in
the `agents` list but do not pass `gym_adapter=…`, the kernel still
detects it but emits a `DeprecationWarning`. Migrate by passing
the gym agent explicitly:

```python
kernel = Kernel(agents=agents, gym_adapter=my_gym_agent, …)
```

The kernel only requires the three-method `GymAdapter` Protocol
(`update_raw_state`, `get_raw_state`, `apply_actions`); any object
that implements them works.

## Pluggable log sink via `LogWriter` (added 2026-04)

The kernel no longer pickles dataframes itself; it delegates to a
`LogWriter` injected via the new `log_writer=` kwarg. Two
implementations ship in `abides_core.log_writer`:

- `NullLogWriter` — writes nothing.
- `BZ2PickleLogWriter(root, run_id)` — the legacy on-disk format
  (`<root>/<run_id>/<name>.bz2`); the run directory is created
  lazily on first write.

If you do not inject a writer, the kernel builds the right one for
you based on `skip_log` and (the new) `log_root` / `log_dir` kwargs,
so existing call sites keep working.

---

## See Also

- `custom-agent-guide.md` — adapter pattern, scaffold, checklist
- `config-system.md` — builder API, templates, oracle config, serialization
- `data-extraction.md` — parsing results, L1/L2 book history
- `parallel-simulation.md` — multiprocessing, RNG hierarchy, log layout

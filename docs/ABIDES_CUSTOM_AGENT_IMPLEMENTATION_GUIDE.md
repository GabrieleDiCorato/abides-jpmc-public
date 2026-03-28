# ABIDES Custom Agent Implementation Guide

This guide is optimized for AI coding agents implementing custom trading strategies in ABIDES.
Since the ABIDES source code is available in your environment, this guide focuses on **architecture, API contracts, and critical behaviors**, avoiding redundant code snippets.

**CRITICAL PREREQUISITE:** Before building any agent, you **MUST** read [`ABIDES_LLM_INTEGRATION_GOTCHAS.md`](./ABIDES_LLM_INTEGRATION_GOTCHAS.md) to understand how market data is asynchronously populated (and avoids `None`/`KeyError` crashes).

---

## 1. Core Architecture & Assumptions

### 1.1 Time & Pricing
- **Time:** Discrete-event simulation. All time variables are integer **nanoseconds**. There are no datetime objects.
- **Prices:** All prices are integer **cents** ($100.00 = 10_000). Never use floats.
- **Holdings:** Cash is tracked in integer cents (`self.holdings["CASH"]`). Asset holdings are integer share counts (`self.holdings["SYMBOL"]`).

### 1.2 Event-Driven Execution (No Loops)
ABIDES agents do not iterate in a loop (no `while True:` or `on_bar()`).
Execution is solely driven by two callbacks:
1. `wakeup(current_time)`: Triggered by the kernel at an agent's scheduled time.
2. `receive_message(current_time, sender_id, message)`: Triggered when a message (market data, order fill, etc.) arrives.

---

## 2. The Recommended Pattern: Interface + Adapter

Do not tightly couple your trading strategy logic directly to ABIDES internals.
Use the **Adapter Pattern** — define your own decoupling layer, then bridge it to ABIDES:

1. **Strategy Protocol (Your Code):** A `typing.Protocol` (or ABC) entirely agnostic of ABIDES. Defines callbacks like `on_tick(snapshot, portfolio)`, `on_fill(fill_info, portfolio)`, `get_wakeup_interval_ns()`. Takes clean dataclasses you define (e.g., `MarketSnapshot`, `PortfolioState`).
2. **Strategy Implementation (Your Code):** Implements the protocol with your trading logic. Zero ABIDES imports.
3. **Adapter (Bridge — Your Code):** A class that inherits from `TradingAgent` and translates ABIDES events into your protocol's callbacks. Registered with the ABIDES config system via `@register_agent`.

> [!NOTE]
> ABIDES does **not** ship a built-in strategy protocol or adapter. You define these in your own project — ABIDES provides the base class (`TradingAgent`) and the config-system hooks (`@register_agent`, `BaseAgentConfig`) to plug them in. This keeps the simulation framework decoupled from any particular strategy interface.

### Adapter skeleton

Your adapter subclasses `TradingAgent` and holds a reference to a strategy instance:

```python
class MyStrategyAdapter(TradingAgent):
    def __init__(self, id, symbol, starting_cash, strategy, *,
                 name=None, type=None, random_state=None,
                 log_orders=False, risk_config=None):
        super().__init__(id, name=name, type=type, random_state=random_state,
                         starting_cash=starting_cash, log_orders=log_orders,
                         risk_config=risk_config)
        self.symbol = symbol
        self.strategy = strategy

    def wakeup(self, current_time):
        if not super().wakeup(current_time):
            return
        # Build a snapshot from ABIDES internals, call strategy.on_tick(...)
        # Translate returned orders → self.place_limit_order / place_market_order
        self.set_wakeup(current_time + self.strategy.get_wakeup_interval_ns())

    def receive_message(self, current_time, sender_id, message):
        super().receive_message(current_time, sender_id, message)
        # Intercept OrderExecutedMsg → call strategy.on_fill(...)
```

### Why extend `TradingAgent`?
`TradingAgent` implements ~1,200 lines of necessary plumbing: exchange discovery, market hours tracking, bid/ask spread caching, portfolio tracking, and order lifecycle management. If you inherit from `Agent` or `FinancialAgent`, you will have to reimplement all of this.

---

## 3. Agent Lifecycle

1. `kernel_initializing(kernel)`: Agent created. Do not send messages.
2. `kernel_starting(start_time)`: Find the `ExchangeAgent`. Schedule the first `wakeup()`.
3. `wakeup(current_time)`: Main active entry point.
   - *CRITICAL:* Always call `super().wakeup(current_time)`. If it returns `False`, the market hours are unknown or the market is closed. **Do not trade if `False`.**
   - Must schedule the next wakeup: `self.set_wakeup(current_time + self.get_wake_frequency())`.
4. `receive_message(current_time, sender_id, message)`: React to inbound data.
   - *CRITICAL:* Always call `super().receive_message(...)` first. `TradingAgent` uses this to update your portfolio and known bounds.
5. `kernel_stopping()`: Log final holdings and PnL. `TradingAgent` automatically marks the portfolio to market.

---

## 4. Retrieving Market Data

ABIDES requires sending a message to request data. The response arrives asynchronously.

### Option A: Subscriptions (Recommended)
Send a subscription request *once* (e.g., in your first valid `wakeup()`).
- Use `L2SubReqMsg`, `TransactedVolSubReqMsg`, etc.
- `TradingAgent` automatically processes `L2DataMsg` and updates `self.known_bids[symbol]` and `self.known_asks[symbol]`.

### Option B: Point-in-time Queries
Send a query (e.g., `self.get_current_spread(symbol)`).
- The result arrives later as a `QuerySpreadResponseMsg`.
- Your `receive_message()` must intercept this to invoke your strategy.

*(Refer to `ABIDES_LLM_INTEGRATION_GOTCHAS.md` for safe access patterns, as all internal data dictionaries start empty).*

---

## 5. Order Management

All orders are placed via `TradingAgent` helpers:
- `self.place_limit_order(symbol, quantity, side, limit_price)`
- `self.place_market_order(symbol, quantity, side)`

**Risk Guards (inherited from TradingAgent):**
- `position_limit` / `position_limit_clamp` — per-symbol position cap (block or clamp).
- `max_drawdown` — if `starting_cash − mark_to_market(holdings) ≥ max_drawdown`, the agent is permanently halted (circuit breaker).
- `max_order_rate` / `order_rate_window_ns` — if more than `max_order_rate` orders are placed within the tumbling window, the agent is permanently halted.

All guards default to `None` (disabled). Set them via the constructor or as `BaseAgentConfig` fields in your config model (see §6). When using the config system, these fields are automatically bundled into a `RiskConfig` object by `BaseAgentConfig._prepare_constructor_kwargs()` and passed to the `TradingAgent` constructor — you do not need to handle this manually.

```python
# Declarative risk guards via config system
config = (SimulationBuilder()
    .from_template("rmsc04")
    .enable_agent("my_strategy", count=1,
                  position_limit=100,         # max 100 shares per symbol
                  max_drawdown=500_000,        # halt if loss >= $5,000
                  max_order_rate=50,           # max 50 orders per window
                  order_rate_window_ns=60_000_000_000)  # 60-second window
    .seed(42)
    .build())
```

**Open Orders Tracking:**
- Monitored in `self.orders` (Dict by `order_id`).
- Automatically decremented/removed by `TradingAgent.order_executed()` when filled.

**Cancellations:**
- Call `self.cancel_order(order)`. This sends a message. The order remains in `self.orders` until the `OrderCancelledMsg` arrives.

---

## 6. Simulation Configuration

Inject your custom agent into an ABIDES simulation using the **declarative config system**. This gives you build-time validation, YAML/JSON serialization, reusable configs, and typed results.

### Step 1: Register your agent

Use `@register_agent` to make your adapter available to the config system. Create a `BaseAgentConfig` subclass that declares your strategy's tunable parameters as Pydantic fields. All `BaseAgentConfig` fields (`starting_cash`, `log_orders`, risk guards, `computation_delay`) are inherited automatically.

```python
from pydantic import Field
from abides_markets.config_system import BaseAgentConfig, register_agent

@register_agent(
    "my_strategy",
    agent_class=MyStrategyAdapter,       # your TradingAgent subclass
    category="strategy",
    description="Custom mean-reversion strategy",
)
class MyStrategyConfig(BaseAgentConfig):
    threshold: float = Field(default=0.05, description="Signal threshold")
    wake_up_freq: str = Field(default="30s", description="Wakeup interval")
```

When `agent_class` is provided, the config system **auto-generates** `create_agents()` by introspecting your adapter's constructor and mapping config field names → constructor parameter names. Fields listed in `_BASE_EXCLUDE` (risk guard fields) are excluded from this mapping — they flow through `RiskConfig` instead.

### Step 2: Override `_prepare_constructor_kwargs()` for computed args

If your adapter needs arguments that aren't simple config-field pass-throughs (e.g., converting a duration string to nanoseconds, or creating a non-serializable strategy instance), override the hook:

```python
class MyStrategyConfig(BaseAgentConfig):
    threshold: float = Field(default=0.05)
    wake_up_freq: str = Field(default="30s")

    def _prepare_constructor_kwargs(self, kwargs, agent_id, agent_rng, context):
        kwargs = super()._prepare_constructor_kwargs(kwargs, agent_id, agent_rng, context)
        from abides_core.utils import str_to_ns
        # Convert string → nanoseconds
        kwargs["wake_up_freq"] = str_to_ns(self.wake_up_freq)
        # Inject a non-serializable strategy object (created fresh per agent)
        kwargs["strategy"] = MyStrategy(
            threshold=self.threshold,
            rng=agent_rng,
        )
        return kwargs
```

> [!IMPORTANT]
> Always call `super()._prepare_constructor_kwargs(...)` first — the base implementation bundles risk guard fields into `RiskConfig`.
>
> Pydantic fields must be JSON-serializable (for YAML/JSON config export). Non-serializable objects like strategy instances must be created inside `_prepare_constructor_kwargs()`, not stored as fields.

### Step 3: Build and run

```python
from abides_markets.config_system import SimulationBuilder
from abides_markets.simulation import run_simulation

config = (SimulationBuilder()
    .from_template("rmsc04")
    .enable_agent("my_strategy", count=1, threshold=0.08)
    .seed(42)
    .build())

result = run_simulation(config)
```

`enable_agent()` accepts any parameters your config model defines. Agent name validation happens at `.build()` time — register your agent **before** calling `.build()`.

The same `SimulationConfig` can be passed to `run_simulation()` any number of times — each call compiles fresh agents, so results are always reproducible.

### Per-agent computation delays

Override how long your agent "thinks" after each wakeup or message:

```python
.enable_agent("my_strategy", count=1, computation_delay=100)  # 100 ns
```

Or set a global default that applies to agents without an override:

```python
.computation_delay(50)   # 50 ns default
```

---

## 7. Custom Oracles — Injecting External Data

If you need to backtest against historical data or generative models (like CGANs) instead of algorithmic mean-reverting data, replace the default oracle.

> [!IMPORTANT]
> The `Oracle` base class is an `abc.ABC`. If you build your own oracle from scratch, you must implement both `get_daily_open_price(self, symbol, mkt_open, cents=True)` and `observe_price(self, symbol, current_time, random_state, sigma_n=1000)`.

See `abides-markets/abides_markets/oracles/external_data_oracle.py` for a full-featured example:
- **Batch mode:** Use `DataFrameProvider` to load full series at initialization.
- **Point mode:** Implement `PointDataProvider` to query a database or generator on demand (uses LRU cache).

### Injecting via the config system

Build the oracle externally and pass it to the builder:

```python
from abides_markets.oracles import ExternalDataOracle, DataFrameProvider

provider = DataFrameProvider({"AAPL": my_historical_series})
oracle = ExternalDataOracle(provider)

config = (SimulationBuilder()
    .oracle_instance(oracle)               # inject pre-built oracle
    .market(ticker="AAPL")
    .enable_agent("noise", count=500)
    .enable_agent("my_strategy", count=1)
    .seed(42)
    .build())
```

For YAML/JSON configs that reference an externally-constructed oracle, use `ExternalDataOracleConfig` as a marker type (`oracle: { type: external_data }`) and pass the oracle instance at compile time: `compile(config, oracle_instance=my_oracle)`.

### Oracle-less simulations

Set `oracle: null` to run without a fundamental-value oracle. This requires `opening_price` (integer cents) and disallows `ValueAgent`:

```python
config = (SimulationBuilder()
    .oracle(type=None)                     # explicitly no oracle
    .market(ticker="ABM", opening_price=100_000)  # $1,000.00
    .enable_agent("noise", count=500)
    .seed(42)
    .build())
```

---

## 8. Running Simulations

### Recommended: `run_simulation()`

```python
from abides_markets.simulation import run_simulation

result = run_simulation(config)              # returns SimulationResult (frozen)
result = run_simulation(config, profile=ResultProfile.QUANT)  # include L1/L2 series
```

`SimulationResult` is an immutable Pydantic model with:
- `result.agents` — list of `AgentData` (per-agent PnL, final holdings, mark-to-market)
- `result.markets` — dict of `MarketSummary` (L1 close, liquidity metrics, optional L1/L2 series)
- `result.metadata` — seed, timing, agent count

Result depth is controlled by `ResultProfile`: `SUMMARY` (default), `QUANT` (adds L1/L2 series), or `FULL` (adds raw agent logs).

### Parallel execution

```python
from abides_markets.simulation import run_batch

configs = [SimulationBuilder().from_template("rmsc04").seed(i).build() for i in range(10)]
results = run_batch(configs, n_workers=4)
```

### Low-level path

For direct Kernel access (e.g., gymnasium environments):

```python
from abides_markets.config_system import compile
from abides_core import abides

runtime = compile(config)          # fresh runtime dict — consumed once
end_state = abides.run(runtime)
# Do NOT reuse `runtime` — call compile() again for another run.
```

---

## 9. Copy-Paste Agent Scaffold

Use this minimal working skeleton as a starting point. Each `# TODO:` marker indicates a line you **must** customise.

```python
"""TODO: module docstring — describe what this agent does."""

from pydantic import Field

from abides_markets.agents import TradingAgent
from abides_markets.config_system import BaseAgentConfig, register_agent
from abides_markets.messages.market import MarketHoursMsg
from abides_core.utils import str_to_ns


# ── Agent class ──────────────────────────────────────────────────────────

class MyAgent(TradingAgent):                      # TODO: rename class
    """TODO: one-line description of strategy behaviour."""

    # Declare every state string your agent uses (enables runtime validation).
    VALID_STATES: frozenset[str] = frozenset({
        "AWAITING_WAKEUP",
        "AWAITING_SPREAD",
        # TODO: add more states as needed
    })

    def __init__(self, id, symbol, starting_cash, *,
                 name=None, type=None, random_state=None,
                 log_orders=False, risk_config=None,
                 wake_up_freq=str_to_ns("30s"),    # TODO: add your params
                 ):
        super().__init__(id, name=name, type=type, random_state=random_state,
                         starting_cash=starting_cash, log_orders=log_orders,
                         risk_config=risk_config)
        self.symbol = symbol                       # REQUIRED — TradingAgent uses it
        self.wake_up_freq = wake_up_freq           # TODO: store your params
        self.state = "AWAITING_WAKEUP"

    def wakeup(self, current_time):
        # Guard: returns False when market hours unknown or market closed.
        if not super().wakeup(current_time):
            return
        self.state = "AWAITING_SPREAD"
        self.get_current_spread(self.symbol)       # async — response arrives later

    def receive_message(self, current_time, sender_id, message):
        # MUST call super first — it updates portfolio, known_bids/asks, etc.
        super().receive_message(current_time, sender_id, message)

        if self.state == "AWAITING_SPREAD":
            bids = self.known_bids.get(self.symbol, [])
            asks = self.known_asks.get(self.symbol, [])
            if bids and asks:
                # TODO: replace with your trading logic
                mid = (bids[0][0] + asks[0][0]) // 2
                self.place_limit_order(self.symbol, 1, "BID", mid)

        # Always schedule the next wakeup.
        self.set_wakeup(current_time + self.wake_up_freq)
        self.state = "AWAITING_WAKEUP"


# ── Config model + registration ──────────────────────────────────────────

@register_agent(
    "my_agent",                                    # TODO: unique registry name
    agent_class=MyAgent,
    category="strategy",                           # TODO: pick category
    description="TODO: one-line description",
)
class MyAgentConfig(BaseAgentConfig):              # TODO: rename class
    wake_up_freq: str = Field(default="30s")
    # TODO: add your config fields here

    def _prepare_constructor_kwargs(self, kwargs, agent_id, agent_rng, context):
        kwargs = super()._prepare_constructor_kwargs(kwargs, agent_id, agent_rng, context)
        kwargs["wake_up_freq"] = str_to_ns(self.wake_up_freq)
        # TODO: inject non-serializable objects here (e.g. strategy instances)
        return kwargs
```

> [!TIP]
> See `abides-markets/abides_markets/agents/noise_agent.py` for a **fully annotated** reference implementation with WHY comments explaining every ABIDES pattern.

---

## 10. Agent-Building Checklist

A step-by-step checklist. Complete every item before considering your agent done.

1. **Subclass `TradingAgent`** — not `Agent` or `FinancialAgent`.
2. **Call `super().__init__(...)`** — pass through `id`, `name`, `type`, `random_state`, `starting_cash`, `log_orders`, `risk_config`. Store `self.symbol`.
3. **Declare `VALID_STATES`** — a `frozenset[str]` of every state your agent uses (enables runtime typo detection).
4. **Guard `wakeup()` entry** — `if not super().wakeup(current_time): return`. Handles unknown market hours and closed market.
5. **Call `super().receive_message(...)` first** — the base class updates portfolio, `known_bids`, `known_asks`, and order tracking.
6. **Schedule the next wakeup** — every code path in `wakeup()` or `receive_message()` that should wake the agent must call `self.set_wakeup(...)`.
7. **Guard `None`/empty data** — `self.known_bids.get(symbol, [])` can be `[]`; `L1DataMsg.bid` can be `None`. Always check before indexing.
8. **Create a `BaseAgentConfig`** — declare tunable parameters as Pydantic fields. Override `_prepare_constructor_kwargs()` for computed or non-serializable args.
9. **Register with `@register_agent`** — provide `agent_class`, `category`, and `description`.
10. **Write a test** — use `make_agent()` from `tests/conftest.py` for unit tests; use `SimulationBuilder` for integration tests (see §11).

---

## 11. Testing Your Agent

### Unit tests with `make_agent()`

`abides-markets/tests/conftest.py` provides `make_agent()` — a helper that constructs any `TradingAgent` subclass with kernel stubs already wired (exchange ID, market hours, RNG, current time). This lets you test agent logic without running a full simulation.

```python
from abides_markets.agents import NoiseAgent
from tests.conftest import make_agent, MKT_OPEN
from abides_core.utils import str_to_ns

def test_initial_state():
    agent = make_agent(NoiseAgent, wakeup_time=MKT_OPEN + str_to_ns("00:05:00"))
    assert agent.state == "AWAITING_WAKEUP"

def test_wakeup_requests_spread(stub_kernel):
    agent = make_agent(NoiseAgent)
    agent.kernel = stub_kernel
    agent.wakeup(agent.current_time)
    assert agent.state == "AWAITING_SPREAD"
```

### Integration tests with `SimulationBuilder`

For end-to-end tests that verify your agent interacts correctly with the exchange, build a short simulation:

```python
from abides_markets.config_system import SimulationBuilder
from abides_markets.simulation import run_simulation

def test_my_agent_runs_in_simulation():
    config = (SimulationBuilder()
        .from_template("rmsc04")
        .market(end_time="09:32:00")   # 2-minute sim for speed
        .enable_agent("my_agent", count=1, wake_up_freq="5s")
        .seed(42)
        .build())
    result = run_simulation(config)
    assert len(result.agents) > 0
```

### When to use each

| Scope | What it tests | Speed | Use `make_agent()` | Use `SimulationBuilder` |
|-------|--------------|-------|--------------------|------------------------|
| **Unit** | State transitions, order placement logic, data guards | Fast (ms) | ✓ | |
| **Integration** | Full lifecycle with exchange, fills, portfolio tracking | Slower (seconds) | | ✓ |

Start with unit tests for rapid iteration, then add one integration test to confirm end-to-end behaviour.

---

## Further Reading

- [`ABIDES_CONFIG_SYSTEM.md`](./ABIDES_CONFIG_SYSTEM.md) — declarative config system, builder, templates, serialization
- [`ABIDES_LLM_INTEGRATION_GOTCHAS.md`](./ABIDES_LLM_INTEGRATION_GOTCHAS.md) — all `None`/`KeyError` traps, safe patterns
- [`ABIDES_DATA_EXTRACTION.md`](./ABIDES_DATA_EXTRACTION.md) — parsing simulation logs and L1/L2 book history
- [`PARALLEL_SIMULATION_GUIDE.md`](./PARALLEL_SIMULATION_GUIDE.md) — multiprocessing, RNG hierarchy, log layout
- [`notebooks/demo_Config_System.ipynb`](../notebooks/demo_Config_System.ipynb) — interactive walkthrough of the config system

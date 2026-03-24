# ABIDES-Markets

**Financial market simulation layer built on ABIDES-Core.**

ABIDES-Markets extends the core discrete-event engine with a realistic
equity-market simulation. It provides a NASDAQ-like exchange agent with a
full limit order book, a library of stylized trading agents (noise, value,
momentum, adaptive market maker, POV execution), fundamental-value oracles,
and a declarative configuration system for composing and running market
simulations. The module targets practitioners and researchers who need a
controllable, agent-based market microstructure laboratory for strategy
back-testing, execution algorithm development, and market design studies.

## Key components

| Component | Description |
|-----------|-------------|
| **ExchangeAgent** | Central market operator. Maintains per-symbol order books, matches limit and market orders (price-time priority), publishes L1/L2/L3 market data, and tracks transacted volume and book imbalance. |
| **TradingAgent** | Base class for all trading strategies. Manages holdings, order lifecycle, market-data subscriptions, and provides convenience methods (`place_order`, `cancel_order`, `get_current_spread`, etc.). |
| **OrderBook** | Price-level–based central limit order book. Supports hidden orders, post-only orders, partial cancellations, and modifications. Provides `get_l1/l2/l3_*_data()` accessors for market-data feeds. |
| **Oracles** | Fundamental-value generators. `SparseMeanRevertingOracle` (Ornstein–Uhlenbeck with megashocks) for synthetic prices; `ExternalDataOracle` with `BatchDataProvider` / `PointDataProvider` for injecting empirical data. |
| **Config system** | Pydantic-based declarative builder (`SimulationBuilder`) with template presets, agent registry (`@register_agent`), YAML/JSON serialization, and a `compile()` step that produces the runtime dict consumed by the Kernel. |

## Public API

```python
from abides_markets import (
    # Agents
    ExchangeAgent, TradingAgent, FinancialAgent,
    NoiseAgent, ValueAgent, MomentumAgent,
    AdaptiveMarketMakerAgent, POVExecutionAgent,
    # Order book
    OrderBook, PriceLevel,
    # Orders
    Order, LimitOrder, MarketOrder, Side,
)
```

## Agents

### NoiseAgent

Random trader. Wakes once, places a single limit order at the current
spread ± a random offset. Provides background liquidity.

### ValueAgent

Fundamental-value trader. Observes a noisy oracle estimate, compares it
with the current market price, and places limit orders around its belief of
fair value. With 10% probability it aggresses the spread. Parameterized by
observation noise `σ_n`, mean reversion `κ`, and shock variance `σ_s`.

### MomentumAgent

Technical-analysis example. Maintains 20- and 50-period moving averages of
the mid-price. Buys when the short MA crosses above the long MA, sells on
the opposite cross. Supports polling or L2 subscription modes.

### AdaptiveMarketMakerAgent

Two-sided ladder market maker following the Chakraborty–Kearns framework.
Sizes orders as a fraction of observed volume (`pov`), anchors the ladder
to the best bid/ask (or mid), and skews inventory with a configurable
`skew_beta`. Adaptive spread widens/narrows with order-book imbalance.

### POVExecutionAgent

Institutional execution algorithm. Targets a percentage of market volume
(`pov`) over a configurable lookback window. Wakes periodically, queries
transacted volume, and places child limit orders until the parent quantity
is filled or time expires.

## Order types

All prices are **integer cents** (`$100.00 = 10_000`).

```python
from abides_markets import LimitOrder, MarketOrder, Side

# Buy 100 shares at $50.00 max
buy = LimitOrder(agent_id=1, time_placed=t, symbol="ABM",
                 quantity=100, side=Side.BID, limit_price=5000)

# Sell 50 shares at market
sell = MarketOrder(agent_id=1, time_placed=t, symbol="ABM",
                   quantity=50, side=Side.ASK)
```

`LimitOrder` supports additional flags: `is_hidden`, `is_post_only`,
`insert_by_id`, and `is_price_to_comply`.

## Oracles

```python
from abides_markets.oracles import SparseMeanRevertingOracle, ExternalDataOracle

# Synthetic OU process
oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)

# Empirical data feed
from abides_markets.oracles.data_providers import BatchDataProvider
oracle = ExternalDataOracle(symbols, data_provider=BatchDataProvider(df))
```

## Configuration and simulation

### Declarative builder (recommended)

```python
from abides_markets.config_system import SimulationBuilder
from abides_markets.simulation import run_simulation

config = SimulationBuilder().from_template("rmsc04").seed(0).build()
result = run_simulation(config)

print(result.metadata)        # seed, timing, tickers
print(result.markets["ABM"])  # per-ticker summary
```

`run_simulation()` compiles a fresh runtime dict internally, runs the
simulation, and returns an immutable `SimulationResult`. The same
`SimulationConfig` can be reused across multiple calls.

### Low-level path

```python
from abides_markets.config_system import SimulationBuilder, compile
from abides_core import abides

config  = SimulationBuilder().from_template("rmsc04").seed(0).build()
runtime = compile(config)   # consumed once
end_state = abides.run(runtime)
```

### Legacy procedural configs

```python
from abides_markets.configs import rmsc04
from abides_core import abides

end_state = abides.run(rmsc04.build_config(seed=0, end_time="10:00:00"))
```

# ABIDES Agent Assessment — Current State, Pain Points & Recommendations

**Date**: 2026-03-20

---

## 1. Agent Inventory

| # | Agent | Category | Registered | File |
|---|-------|----------|:---:|------|
| 1 | **ExchangeAgent** | Infrastructure | N/A | `abides-markets/abides_markets/agents/exchange_agent.py` |
| 2 | **NoiseAgent** | Background | `"noise"` | `abides-markets/abides_markets/agents/noise_agent.py` |
| 3 | **ValueAgent** | Background | `"value"` | `abides-markets/abides_markets/agents/value_agent.py` |
| 4 | **MomentumAgent** | Strategy | `"momentum"` | `abides-markets/abides_markets/agents/examples/momentum_agent.py` |
| 5 | **AdaptiveMarketMakerAgent** | Market Maker | `"adaptive_market_maker"` | `abides-markets/abides_markets/agents/market_makers/adaptive_market_maker_agent.py` |
| 6 | **POVExecutionAgent** | Execution | `"pov_execution"` | `abides-markets/abides_markets/agents/pov_execution_agent.py` |
| 7 | **CoreBackgroundAgent** | Gym Base | No | `abides-markets/abides_markets/agents/background/core_background_agent.py` |
| 8 | **FinancialGymAgent** | Gym | No | `abides-gym/abides_gym/experimental_agents/financial_gym_agent.py` |

**Total: 6 concrete trading/infrastructure agents + 2 gym support agents.**

---

## 2. Per-Agent Assessment

### 2.1 ExchangeAgent (962 LOC)

**Role**: Central order book, matching engine, subscription publisher.

| Concern | Severity | Detail |
|---------|:--------:|--------|
| `deepcopy` on every limit order | Medium | Line 590 — every inbound `LimitOrderMsg` is deep-copied before book insertion. With 100s of agents sending 1000s of orders this is a non-trivial GC burden. |
| No multi-symbol matching parallelism | Low | Orders for independent symbols are processed sequentially in the same `receive_message` call. |
| `isinstance` dispatch chain | Low | ~20 `isinstance` checks per message. A dispatch dict would be cleaner and marginally faster. |
| TODO at line 373, 894 | Low | Acknowledged tech debt in the message hierarchy and order type categorization. |

### 2.2 NoiseAgent

**Role**: Wakes once, places a single random limit order at bid or ask.

| Concern | Severity | Detail |
|---------|:--------:|--------|
| Single-shot agent | Design | Places exactly **one** order then goes silent. In real markets noise traders interact continuously. A configurable multi-wake mode would produce more realistic microstructure. |
| `kernel_stopping` surplus calculation | Low | Lines 80-100: mixes integer cents with float division (`float(surplus) / self.starting_cash`), which is fine functionally but inconsistent with the "prices are integer cents" convention. |
| `type(self) is NoiseAgent` guard | Code smell | Line 136 — fragile isinstance-vs-type check prevents subclass reuse. |

### 2.3 ValueAgent

**Role**: Bayesian fundamental-value estimator. Core of price discovery.

| Concern | Severity | Detail |
|---------|:--------:|--------|
| **Type annotation bug** | ~~Low~~ | ~~Line 29: `log_orders: float = False` — should be `bool`.~~ **Fixed in v2.0.0.** |
| `cancel_all_orders()` then immediately re-queries | Medium | Lines 132-138: on every wakeup the agent cancels all open orders and re-places them. This generates 2x message traffic (cancel + new order) instead of amending. With many ValueAgents this multiplies exchange load. |
| ~~`buy` variable conflation~~ | ~~Low~~ | ~~Line 245: `buy` is set to `True`/`False` in the `if/elif` branches but to `self.random_state.randint(0, 1 + 1)` (integer 0 or 1) in the `else`. Then `Side.BID if buy == 1` — works but is fragile mixed-type logic.~~ **Fixed in v2.0.0** — normalised to `bool` with `bool(self.random_state.randint(0, 2))`. |
| `type(self) is ValueAgent` guard | Code smell | Same pattern as NoiseAgent. |
| Hard-coded `depth_spread = 2` | Low | Not configurable via constructor or config system. |

### 2.4 MomentumAgent

**Role**: Moving-average crossover (20 vs 50 bar).

| Concern | Severity | Detail |
|---------|:--------:|--------|
| **Unsafe `known_bids` access in subscribe mode** | ~~High~~ | ~~Line 101: `self.known_bids[self.symbol]` — will raise `KeyError` if the first message arrives before any spread data. Should use `.get(symbol, [])`.~~ **Fixed in v2.0.0.** |
| Unbounded `mid_list` growth | Medium | Line 112: `self.mid_list.append(...)` grows forever. In a 6.5-hour trading day at 1s frequency = 23,400 entries. `ma()` recomputes `np.cumsum` over the full array each time — O(n) growing cost. Should use a rolling window / `deque`. |
| Hard-coded MA windows (20, 50) | Design | Not configurable. A production momentum agent needs tunable fast/slow windows. |
| No position management | Design | Keeps buying/selling without any position limit or reversal logic. Can accumulate unbounded inventory. |

### 2.5 AdaptiveMarketMakerAgent

**Role**: Ladder market-maker with inventory skew and adaptive spread.

| Concern | Severity | Detail |
|---------|:--------:|--------|
| **Bug: wrong state key in subscribe mode** | ~~Critical~~ | ~~Line 326: `self.state["MARKET_DATA"]` should be `self.state["AWAITING_MARKET_DATA"]`.~~ **Fixed in v2.0.0.** |
| Cancel-then-repost strategy | Medium | Every wakeup calls `cancel_all_orders()` then places a fresh ladder. This is 2 × `num_ticks` cancellation messages + 2 × `num_ticks` new orders. Amend-in-place would halve message volume. |
| No P&L / risk tracking | Design | No max-position limit, no loss threshold, no end-of-day flatten. A production MM needs all of these. |
| `subscribe_freq` as `float` | Low | Declared as `float` but used as `int` in ns comparisons. |

### 2.6 POVExecutionAgent

**Role**: Percentage-of-volume execution algorithm.

| Concern | Severity | Detail |
|---------|:--------:|--------|
| ~~`last_bid`/`last_ask` typed as `Optional[float]`~~ | ~~Low~~ | ~~Lines 132-133: prices should be `Optional[int]` (integer cents).~~ **Fixed in v2.0.0** — corrected to `Optional[int]`. |
| Not tracking execution fill prices | Medium | `order_executed` tracks quantity but not price. Cannot compute VWAP of executed fills for slippage analysis. |
| Market orders only | Design | Uses exclusively market orders for fills. A production POV agent should also support limit orders with active crossing for better price improvement. |
| No urgency parameter | Design | Real POV algos have urgency controls (e.g., ramp rate near close). |

### 2.7 TradingAgent Base (1,250 LOC)

| Concern | Severity | Detail |
|---------|:--------:|--------|
| `deepcopy` on every order placement | Medium | Lines 556, 611, 647: every `place_limit_order`/`place_market_order` deep-copies the order. Performance cost scales linearly with order flow. |
| ~~`mark_to_market` KeyError risk~~ | ~~Medium~~ | ~~Lines 1160/1164: `self.last_trade[symbol]` will raise `KeyError` if no trade has occurred for a held symbol.~~ **Fixed in v2.0.0** — replaced with safe `.get(symbol)` + None guard. |
| Self-comment: "ugly way" at line 197 | Low | Result tracking via kernel dict is acknowledged tech debt. |

---

## 3. Systemic Pain Points

### 3.1 Missing Agent Types (Critical Gaps for Production)

The current agent roster is **minimal for academic simulation** but **incomplete for a deployable market simulation product**:

| Gap | Impact | Priority |
|-----|--------|:--------:|
| **TWAP Execution Agent** | Cannot simulate time-weighted execution — the most basic institutional algo. | **P0** |
| **VWAP Execution Agent** | Cannot simulate volume-weighted execution — industry standard alongside TWAP. The existing POV agent is close but structurally different (it targets participation rate, not price benchmark). | **P0** |
| **Implementation Shortfall (Arrival Price) Agent** | No Almgren-Chriss style optimal execution. Critical for measuring execution quality. | **P1** |
| **Iceberg / Hidden Agent** | No agent that uses hidden or iceberg order types despite the exchange supporting `is_hidden`. | **P1** |
| **Stop-Loss / Take-Profit Agent** | No conditional order logic. Real markets have significant stop-triggered volume. | **P1** |
| **Informed Trader / News Agent** | A step beyond ValueAgent — an agent that receives discrete information shocks (earnings, events) and trades aggressively on them. Essential for studying adverse selection. | **P1** |
| **Pairs / Stat-Arb Agent** | No multi-symbol agent despite ExchangeAgent supporting multiple symbols. Needed for any relative-value simulation. | **P2** |
| **HFT / Latency-Sensitive Agent** | No agent designed to exploit microsecond-level speed advantages (queue priority, stale quotes). | **P2** |
| **Mean-Reversion Agent** | An alternative to MomentumAgent for modeling contrarian strategies. | **P2** |

### 3.2 Performance Bottlenecks

1. **`deepcopy` proliferation**: The `TradingAgent.place_limit_order` → `deepcopy(order)` and `ExchangeAgent` → `deepcopy(message.order)` paths mean every order is deep-copied **twice** (once by sender, once by exchange). With 1000 agents × 100 orders/day = 200,000 unnecessary deep copies.

2. **Cancel-and-repost pattern**: Both `AdaptiveMarketMakerAgent` and `ValueAgent` cancel all orders, then re-place. This generates 4x the message count vs. order amendment. At scale this dominates kernel event-queue processing time.

3. **`isinstance` dispatch chains** in `ExchangeAgent.receive_message` (15+ checks) and `TradingAgent.receive_message` (12+ checks) — on every single message. A `dict`-based dispatch would be O(1).

### 3.3 Robustness Issues

1. **No position limits anywhere**: Not a single agent enforces a max position. The `ignore_risk=True` default means even the risk check is bypassed by default.

2. **No agent-level circuit breaker**: If a strategy enters a pathological loop (e.g., MomentumAgent accumulating 100,000 shares), nothing stops it.

3. **~~Unsafe dictionary access patterns~~**: ~~Multiple agents access `self.known_bids[symbol]` directly instead of `.get(symbol, [])`, risking `KeyError` at runtime.~~ **Fixed in v2.0.0** — `TradingAgent.get_known_bid_ask_midpoint()` and `TradingAgent.mark_to_market()` now use safe `.get()` access.

---

## 4. Recommendations

### 4.1 Immediate Fixes (Bugs)

| Fix | File | Line | Status |
|-----|------|------|:------:|
| Change `self.state["MARKET_DATA"]` → `self.state["AWAITING_MARKET_DATA"]` | `adaptive_market_maker_agent.py` | 326 | ✅ v2.0.0 |
| Change `log_orders: float` → `log_orders: bool` | `value_agent.py` | 29 | ✅ v2.0.0 |
| Use `.get(symbol, [])` in MomentumAgent subscribe mode | `momentum_agent.py` | 101 | ✅ v2.0.0 |
| Fix `get_known_bid_ask()` return type `float` → `int` | `trading_agent.py` | 1078 | ✅ v2.0.0 |
| Fix `get_known_bid_ask_midpoint()` bare dict KeyError | `trading_agent.py` | 1199 | ✅ v2.0.0 |
| Fix `mark_to_market()` bare dict KeyError (×2) | `trading_agent.py` | 1160/1168 | ✅ v2.0.0 |
| Remove dead `if log_orders is None:` block | `trading_agent.py` | 85 | ✅ v2.0.0 |
| Fix duplicate `isinstance` L3 check | `exchange_agent.py` | 758 | ✅ v2.0.0 |
| Init `metric_trackers` when `use_metric_tracker=False` | `exchange_agent.py` | 191 | ✅ v2.0.0 |
| Normalise `buy` variable to `bool` | `value_agent.py` | 256 | ✅ v2.0.0 |
| Normalise `buy_indicator` → `buy` as `bool` | `noise_agent.py` | 142 | ✅ v2.0.0 |
| Fix MA values stored as floats (`.round(2)` → `int(round(...))`) | `momentum_agent.py` | 112 | ✅ v2.0.0 |
| Fix discarded `initialise_state()` return value | `adaptive_market_maker_agent.py` | 239 | ✅ v2.0.0 |
| Fix `last_bid`/`last_ask` type `float` → `int` | `pov_execution_agent.py` | 131 | ✅ v2.0.0 |

### 4.2 New Agent Types to Implement

**Tier 1 — Required for product launch:**

| Agent | Description |
|-------|-------------|
| **TWAPExecutionAgent** | Slices a large parent order into uniform time-sliced child orders. Configurable start/end time, randomization interval, limit-vs-market child orders. |
| **VWAPExecutionAgent** | Targets the volume-weighted average price by distributing execution according to a historical volume profile curve. |
| **InformedTraderAgent** | Receives discrete information shocks from the oracle and trades aggressively in the correct direction. Parametrize by information precision and aggressiveness. |

**Tier 2 — Required for realistic microstructure:**

| Agent | Description |
|-------|-------------|
| **IcebergAgent** | Places large orders using iceberg/hidden order types to minimize information leakage. |
| **StopLossAgent** | Monitors price and converts conditional stop/take-profit orders into market orders when triggered. |
| **ImplementationShortfallAgent** | Almgren-Chriss optimal execution agent that balances market impact vs. timing risk. |

**Tier 3 — Differentiation:**

| Agent | Description |
|-------|-------------|
| **PairsArbitrageAgent** | Trades two correlated symbols based on spread z-score. First multi-symbol agent. |
| **MeanReversionAgent** | Buys when price is N standard deviations below recent mean, sells above. Contrarian complement to MomentumAgent. |
| **HFTAgent** | Exploits queue priority and latency; cancels stale quotes immediately on price change. |

### 4.3 Architectural Improvements

1. **Message dispatch refactor**: Replace `isinstance` chains with `dict[type, Callable]` dispatch in `TradingAgent.receive_message` and `ExchangeAgent.receive_message`. Expected ~15-20% speedup on message handling hot path.

2. **Order amendment support**: Add `amend_order()` to `TradingAgent` that modifies price/quantity in-place on the exchange, rather than cancel+replace. This would halve message count for market makers.

3. **Position limit mixin**: A `PositionLimitMixin` that any agent can inherit to enforce hard position caps, with configurable breach behavior (block order / flatten).

4. **Rolling window utility**: Replace the unbounded list accumulation in MomentumAgent (and future signal agents) with a `deque(maxlen=N)` + incremental statistic calculation.

5. **Register all agents**: The gym agents (`CoreBackgroundAgent`, `FinancialGymAgent`) are not registered in the config system. For deployment, every agent type should be configurable declaratively.

---

## 5. Oracle System Assessment

### 5.1 Oracle Inventory

| # | Oracle | Status | File |
|---|--------|:------:|------|
| 1 | **Oracle** (ABC) | Base class | `abides-markets/abides_markets/oracles/oracle.py` |
| 2 | **MeanRevertingOracle** | Legacy / dangerous | `abides-markets/abides_markets/oracles/mean_reverting_oracle.py` |
| 3 | **SparseMeanRevertingOracle** | Default | `abides-markets/abides_markets/oracles/sparse_mean_reverting_oracle.py` |
| 4 | **ExternalDataOracle** | Active | `abides-markets/abides_markets/oracles/external_data_oracle.py` |

**Exports**: `__init__.py` exports `SparseMeanRevertingOracle` and `ExternalDataOracle`. `MeanRevertingOracle` is **not** in `__all__` but is still importable (and configurable via `MeanRevertingOracleConfig`).

### 5.2 API Contract

The `Oracle` ABC defines two abstract methods:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `get_daily_open_price` | `(symbol, mkt_open, cents=True) → int` | Returns the fundamental open price in integer cents. Used by `ExchangeAgent` to seed `OrderBook.last_trade`. |
| `observe_price` | `(symbol, current_time, random_state, sigma_n=1000) → int` | Returns a noisy observation of the current fundamental value. `sigma_n=0` gives the exact value. Used by `ValueAgent` for Bayesian updates. |

**Gap**: No `f_log` property in the ABC. `ExchangeAgent.kernel_stopping()` (line 274) uses `hasattr(self.oracle, "f_log")` to conditionally write the fundamental value log. This means `ExternalDataOracle` silently skips fundamental logging.

### 5.3 Oracle Injection Path

```
SimulationConfig.market.oracle  (Pydantic model)
    → compiler._build_oracle()  (instantiation)
    → runtime["custom_properties"]["oracle"]
    → Kernel.custom_properties["oracle"]
    → Kernel.oracle  (property)
    → Agent.kernel_starting(): self.oracle = self.kernel.oracle
```

Agents access the oracle via `self.kernel.oracle` in their `kernel_starting()` override. The oracle is shared across all agents (singleton per simulation).

### 5.4 Agent–Oracle Coupling Map

| Agent | Stores Ref | Calls Methods | Methods Used |
|-------|:----------:|:-------------:|--------------|
| **ExchangeAgent** | ✅ (line 225) | ✅ | `get_daily_open_price` (line 230), `f_log` via `hasattr` (line 274) |
| **ValueAgent** | ✅ (line 74) | ✅ | `observe_price(sigma_n=self.sigma_n)` for noisy Bayesian update (line 155); `observe_price(sigma_n=0)` for exact final valuation (line 85) |
| **NoiseAgent** | ✅ (line 65) | ❌ | **Dead reference** — stores `self.oracle = self.kernel.oracle` but never calls any method on it |
| **MomentumAgent** | ❌ | ❌ | No oracle access |
| **AdaptiveMarketMakerAgent** | ❌ | ❌ | No oracle access |
| **POVExecutionAgent** | ❌ | ❌ | No oracle access |

**Key finding**: Only 1 of 5 trading agents (ValueAgent) actually uses the oracle for trading decisions. The oracle is architecturally central but practically underutilized.

### 5.5 Per-Oracle Assessment

#### 5.5.1 MeanRevertingOracle (Legacy)

| Concern | Severity | Detail |
|---------|:--------:|--------|
| **OOM on real-time-scale simulations** | Critical | Line 89: `pd.date_range(mkt_open, mkt_close, freq="ns")` generates a nanosecond-resolution Series. For a 6.5h trading day: 6.5 × 3,600 × 10⁹ = 2.34×10¹³ entries → instant OOM. Only safe when `mkt_close - mkt_open` is < ~10⁶ ns. |
| Still compilable | Medium | `MeanRevertingOracleConfig` is accepted by the compiler (line 200), so users can accidentally trigger the OOM via declarative config. |
| No deprecation warning | ~~Low~~ | ~~Silently constructs without warning users to prefer `SparseMeanRevertingOracle`.~~ **Fixed in v2.0.0** — `DeprecationWarning` + step-count guard added. |

#### 5.5.2 SparseMeanRevertingOracle (Default)

| Concern | Severity | Detail |
|---------|:--------:|--------|
| On-demand computation | Strength | Stores only `(timestamp, value)` per symbol. Computes OU process forward lazily — safe for any time scale. |
| Per-symbol RandomState isolation | Strength | Sub-allocates independent random states per symbol, ensuring reproducibility even when agent count changes. |
| Megashock system | Strength | Poisson-arrival bimodal shocks create realistic price discontinuities beyond simple OU noise. |
| `f_log` unbounded growth | Medium | `self.f_log[symbol].append(...)` on every `advance_fundamental_value_series()` call. In a busy simulation this list grows without bound. |
| No megashock visibility to agents | Design | Agents cannot detect or react to specific megashock events. An InformedTraderAgent pattern would require an event subscription mechanism. |
| No per-symbol correlation | Design | Each symbol's OU process is fully independent. Unrealistic for correlated assets (e.g., sector peers). |

#### 5.5.3 ExternalDataOracle

| Concern | Severity | Detail |
|---------|:--------:|--------|
| Clean provider protocol design | Strength | `BatchDataProvider` / `PointDataProvider` are structural-typing Protocols — no subclassing required. |
| Three interpolation strategies | Strength | FORWARD_FILL, NEAREST, LINEAR — covers all common use cases for timestamp gaps. |
| LRU cache for point mode | Strength | Configurable `cache_size` keeps memory bounded for database-backed lookups. |
| **Not compilable via config system** | Critical | `compiler.py` line 213 raises `NotImplementedError` for `ExternalDataOracleConfig`. The most production-relevant oracle cannot be configured declaratively. Workaround: manual injection via `custom_properties`, which bypasses the config system's validation. |
| **No `f_log` implementation** | ~~Medium~~ | ~~`ExchangeAgent.kernel_stopping()` silently skips fundamental value logging when using this oracle, since it lacks `f_log`.~~ **Fixed in v2.0.0** — Oracle ABC now provides default empty `f_log`. |

### 5.6 Data Providers

| Provider | Type | Description |
|----------|------|-------------|
| `BatchDataProvider` | Protocol | `get_fundamental_series(symbol, start, end) → pd.Series` — returns full series at once. Best for file-based data. |
| `PointDataProvider` | Protocol | `get_fundamental_at(symbol, timestamp) → int` — returns single value per call. Best for DB queries, generative models. |
| `DataFrameProvider` | Concrete | Reference `BatchDataProvider` wrapping `Dict[str, pd.Series]`. Slices by timestamp range. |

**Gap**: No built-in CSV or Parquet provider. Users must write their own loader. A `CsvProvider` and `ParquetProvider` would cover the two most common external data formats.

### 5.7 Oracle System Pain Points

1. **~~`f_log` not in ABC~~**: ~~`ExchangeAgent` relies on `hasattr(self.oracle, "f_log")` (line 274). This fragile duck-typing means fundamental value logging is silently skipped for `ExternalDataOracle` and any future oracle that omits `f_log`.~~ **Fixed in v2.0.0** — `f_log` is now a class attribute on the Oracle ABC defaulting to `{}`; `ExchangeAgent` uses a truthiness check.

2. **~~NoiseAgent dead oracle reference~~**: ~~Line 65 stores `self.oracle = self.kernel.oracle` but never calls any method — wastes memory and misleads developers reading the code.~~ **Fixed in v2.0.0** — dead line removed.

3. **Oracle severely underutilized**: Only `ValueAgent` uses oracle observations for trading decisions (2 of 6 agents reference oracle, but NoiseAgent's reference is dead). Agents like `MomentumAgent` and `AdaptiveMarketMakerAgent` have no fundamental value anchor, limiting their behavioral realism.

4. **ExternalDataOracle not compilable**: The `NotImplementedError` at `compiler.py:213` blocks the most production-relevant oracle from declarative configuration. This forces users into manual `compile()` + injection workflows.

5. **~~MeanRevertingOracle OOM risk~~**: ~~Still configurable via `MeanRevertingOracleConfig` despite producing instant OOM for any realistic time scale simulation. No guard or deprecation warning.~~ **Fixed in v2.0.0** — `DeprecationWarning` issued on instantiation; `ValueError` raised if step count > 1,000,000.

6. **No oracle event subscription API**: Agents cannot subscribe to discrete information events (megashocks, earnings announcements). The megashock system in `SparseMeanRevertingOracle` is purely internal — agents cannot react to specific exogenous shocks, preventing implementation of informed trader strategies.

---

## 6. Oracle Recommendations

### 6.1 Immediate Fixes

| Fix | File | Detail | Status |
|-----|------|--------|:------:|
| Add `f_log` class attribute to Oracle ABC | `oracle.py` | Default to `{}` (empty dict); override in concrete oracles. Eliminates fragile `hasattr` check in ExchangeAgent. | ✅ v2.0.0 |
| Remove dead `self.oracle` in NoiseAgent | `noise_agent.py:65` | Delete `self.oracle = self.kernel.oracle`. | ✅ v2.0.0 |
| Implement `ExternalDataOracleConfig` compilation | `compiler.py:212-217` | Load CSV/Parquet from `data_path`, construct `DataFrameProvider`, pass to `ExternalDataOracle`. | — |

### 6.2 Safety Improvements

| Improvement | Detail | Status |
|-------------|--------|:------:|
| Step-count guard on MeanRevertingOracle | Raise `ValueError` if `mkt_close - mkt_open` exceeds a safe threshold (e.g., 10⁶ steps). Prevents accidental OOM. | ✅ v2.0.0 |
| Deprecation warning on MeanRevertingOracle | Log a `DeprecationWarning` when instantiated, directing users to `SparseMeanRevertingOracle`. | ✅ v2.0.0 |
| Bounded `f_log` growth | Use `deque(maxlen=N)` or periodic flush to prevent unbounded memory growth in `SparseMeanRevertingOracle.f_log`. | — |

### 6.3 Strategic Enhancements

| Enhancement | Priority | Detail |
|-------------|:--------:|--------|
| Oracle event subscription API | P1 | Allow agents to register for discrete information shocks (megashocks, earnings). Enables the `InformedTraderAgent` pattern described in §4.2. |
| `ValueAgentConfig` auto-derive `r_bar` | P1 | `_prepare_constructor_kwargs()` should extract `r_bar` from the oracle config, eliminating parameter duplication and misalignment risk. |
| Multi-symbol correlation | P2 | Current oracles generate independent series per symbol. Add covariance structure (e.g., Cholesky-decomposed correlated OU processes) for realistic cross-asset modeling. |
| Built-in CSV/Parquet providers | P2 | Ship `CsvProvider` and `ParquetProvider` alongside `DataFrameProvider` to cover the most common external data ingestion patterns without user boilerplate. |

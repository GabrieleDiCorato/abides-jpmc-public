# ABIDES Agent & Product Assessment

**Date**: 2026-03-26 (updated)

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


### 2.2 NoiseAgent

**Role**: Wakes once, places a single random limit order at bid or ask.

| Concern | Severity | Detail |
|---------|:--------:|--------|
| Single-shot agent | Design | Places exactly **one** order then goes silent. In real markets noise traders interact continuously. A configurable multi-wake mode would produce more realistic microstructure. |

### 2.4 MomentumAgent

**Role**: Moving-average crossover with configurable short/long windows.

| Concern | Severity | Detail |
|---------|:--------:|--------|
| No position management | Design | Keeps buying/selling without any position limit or reversal logic. Can accumulate unbounded inventory. |

### 2.5 AdaptiveMarketMakerAgent

**Role**: Ladder market-maker with inventory skew and adaptive spread.

| Concern | Severity | Detail |
|---------|:--------:|--------|
| No P&L / risk tracking | Design | No max-position limit, no loss threshold, no end-of-day flatten. A production MM needs all of these. |

### 2.6 POVExecutionAgent

**Role**: Percentage-of-volume execution algorithm.

| Concern | Severity | Detail |
|---------|:--------:|--------|
| Market orders only | Design | Line 359: uses exclusively market orders for fills. A production POV agent should also support limit orders with active crossing for better price improvement. |
| No urgency parameter | Design | Real POV algos have urgency controls (e.g., ramp rate near close). |


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
| **Informed Trader / News Agent** | Distinct from ValueAgent's continuous Bayesian estimation — reacts to **discrete** oracle events (megashocks, earnings) with urgency. Essential for studying adverse selection: when informed flow hits the book, MM spreads widen and liquidity dries up. Requires oracle event subscription API (§6.2). | **P1** |
| **Pairs / Stat-Arb Agent** | No multi-symbol agent despite ExchangeAgent supporting multiple symbols. Needed for any relative-value simulation. | **P2** |
| **HFT / Latency-Sensitive Agent** | No agent designed to exploit microsecond-level speed advantages (queue priority, stale quotes). | **P2** |
| **Mean-Reversion Agent** | An alternative to MomentumAgent for modeling contrarian strategies. | **P2** |

### 3.2 Robustness Issues


1. ~~**No agent-level circuit breaker**~~: **RESOLVED.** Implemented directly in `TradingAgent`. Constructor params: `max_drawdown: int | None` (loss-from-starting-cash threshold in cents), `max_order_rate: int | None` (max orders per window), `order_rate_window_ns: int` (tumbling window, default 1 min). Once either trigger fires the latch is permanent — the agent is halted for the remainder of the simulation. `BaseAgentConfig` fields propagate via the config system.

---

## 4. Recommendations

### 4.1 New Agent Types to Implement

**Tier 1 — Required for product launch:**

| Agent | Description |
|-------|-------------|
| **TWAPExecutionAgent** | Slices a large parent order into uniform time-sliced child orders. Configurable start/end time, randomization interval, limit-vs-market child orders. |
| **VWAPExecutionAgent** | Targets the volume-weighted average price by distributing execution according to a historical volume profile curve. |
| **InformedTraderAgent** | Receives **discrete** information shocks (megashocks, earnings events) from the oracle via event subscription and trades aggressively in the correct direction. Differs from `ValueAgent` (continuous Bayesian observation) by reacting to *events* with urgency. Key parameters: `information_delay` (0ns insider → minutes slow analyst), `information_precision` (perfect → noisy), `aggressiveness` (fraction of edge consumed per trade). Models adverse selection — when this agent hits the book, AMM spreads should widen and liquidity should dry up. Requires oracle event subscription API (§6.2). |

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

### 4.2 Architectural Improvements

1. ~~**Position limit mixin**~~: **RESOLVED.** Implemented directly in `TradingAgent` (no mixin needed — all trading agents inherit it). Constructor params: `position_limit: int | None` (symmetric `[-N, +N]`), `position_limit_clamp: bool` (clamp vs. block). Pending orders are conservatively counted. `BaseAgentConfig` fields propagate via the config system.

2. **Register all agents**: The gym agents (`CoreBackgroundAgent`, `FinancialGymAgent`) are not registered in the config system. For deployment, every agent type should be configurable declaratively.

---

## 5. Oracle System Assessment

### 5.1 Oracle Inventory

| # | Oracle | Status | File |
|---|--------|:------:|------|
| 1 | **Oracle** (ABC) | Base class | `abides-markets/abides_markets/oracles/oracle.py` |
| 2 | **MeanRevertingOracle** | Deprecated | `abides-markets/abides_markets/oracles/mean_reverting_oracle.py` |
| 3 | **SparseMeanRevertingOracle** | Default | `abides-markets/abides_markets/oracles/sparse_mean_reverting_oracle.py` |
| 4 | **ExternalDataOracle** | Active | `abides-markets/abides_markets/oracles/external_data_oracle.py` |

**Exports**: `__init__.py` exports `SparseMeanRevertingOracle` and `ExternalDataOracle`. `MeanRevertingOracle` is **not** in `__all__` but is still importable (and configurable via `MeanRevertingOracleConfig`).

### 5.2 API Contract

The `Oracle` ABC defines two abstract methods:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `get_daily_open_price` | `(symbol, mkt_open, cents=True) → int` | Returns the fundamental open price in integer cents. Used by `ExchangeAgent` to seed `OrderBook.last_trade`. |
| `observe_price` | `(symbol, current_time, random_state, sigma_n=1000) → int` | Returns a noisy observation of the current fundamental value. `sigma_n=0` gives the exact value. Used by `ValueAgent` for Bayesian updates. |

The ABC also provides a class-level `f_log: dict[str, list[dict[str, Any]]] = {}` attribute. Subclasses that track fundamental history override it in `__init__`; those that don't inherit the empty default. `ExchangeAgent` uses a truthiness check (`if self.oracle.f_log:`) to conditionally write the fundamental value log.

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
| **ExchangeAgent** | ✅ (line 225) | ✅ | `get_daily_open_price` (line 230), `f_log` truthiness check (line 275) |
| **ValueAgent** | ✅ (line 74) | ✅ | `observe_price(sigma_n=self.sigma_n)` for noisy Bayesian update (line 155); `observe_price(sigma_n=0)` for exact final valuation (line 85) |
| **NoiseAgent** | ❌ | ❌ | No oracle access (dead reference removed in v2.0.0) |
| **MomentumAgent** | ❌ | ❌ | No oracle access |
| **AdaptiveMarketMakerAgent** | ❌ | ❌ | No oracle access |
| **POVExecutionAgent** | ❌ | ❌ | No oracle access |

**Key finding**: Only `ValueAgent` uses the oracle for trading decisions — and this is **correct by design**. The separation between informed agents (ValueAgent → oracle → limit orders near fundamental) and uninformed agents (everyone else → LOB-only) models realistic information asymmetry. In a typical rmsc04 simulation, 102 ValueAgents collectively push LOB prices toward fundamentals; MomentumAgent, AMM, NoiseAgent, and POV then react to those prices — exactly like their real-world counterparts who lack private fundamental models. This heterogeneity IS the price discovery mechanism.

**Oracle access rationale per agent**:

| Agent | Real-world analogue | Oracle access? | Rationale |
|-------|---------------------|:--------------:|----------|
| **ValueAgent** | Fundamental analyst | ✅ Yes | Bayesian estimation of fundamental value — the canonical informed trader. |
| **MomentumAgent** | Trend-following CTA | ❌ No | Momentum strategies are purely price-based by definition. Adding oracle access would destroy the informed/uninformed heterogeneity. |
| **AdaptiveMarketMakerAgent** | Electronic MM | ❌ No | Most MMs quote around LOB mid, not fundamentals. An optional fundamental anchor could be a future enhancement (see §6.2). |
| **NoiseAgent** | Retail/random flow | ❌ No | Axiomatically uninformed — oracle access would contradict the agent's role. |
| **POVExecutionAgent** | Execution desk algo | ❌ No | Benchmark-targeting algorithm, not alpha-generating. Execution agents should not have fundamental views. |

### 5.5 Per-Oracle Assessment

#### 5.5.1 MeanRevertingOracle (Deprecated)

| Concern | Severity | Detail |
|---------|:--------:|--------|
| Still compilable | Medium | `MeanRevertingOracleConfig` is accepted by the compiler, so users can accidentally reach the step-count guard via declarative config. Consider removing the config model entirely to eliminate the foot-gun. |

*v2.0.0 added a `DeprecationWarning` and a `ValueError` guard rejecting > 1,000,000 steps, effectively neutering the OOM risk.*

#### 5.5.2 SparseMeanRevertingOracle (Default)

| Concern | Severity | Detail |
|---------|:--------:|--------|
| On-demand computation | Strength | Stores only `(timestamp, value)` per symbol. Computes OU process forward lazily — safe for any time scale. |
| Per-symbol RandomState isolation | Strength | Sub-allocates independent random states per symbol, ensuring reproducibility even when agent count changes. |
| Megashock system | Strength | Poisson-arrival bimodal shocks create realistic price discontinuities beyond simple OU noise. |
| No megashock visibility to agents | Design | Agents cannot detect or react to specific megashock events. An InformedTraderAgent pattern would require an event subscription mechanism. |
| No per-symbol correlation | Design | Each symbol's OU process is fully independent. Unrealistic for correlated assets (e.g., sector peers). |

#### 5.5.3 ExternalDataOracle

| Concern | Severity | Detail |
|---------|:--------:|--------|
| Clean provider protocol design | Strength | `BatchDataProvider` / `PointDataProvider` are structural-typing Protocols — no subclassing required. |
| Three interpolation strategies | Strength | FORWARD_FILL, NEAREST, LINEAR — covers all common use cases for timestamp gaps. |
| LRU cache for point mode | Strength | Configurable `cache_size` keeps memory bounded for database-backed lookups. |
| **Not compilable via config system** | Critical | `compiler.py` line 213 raises `NotImplementedError` for `ExternalDataOracleConfig`. The most production-relevant oracle cannot be configured declaratively. Workaround: manual injection via `custom_properties`, which bypasses the config system's validation. |

### 5.6 Data Providers

| Provider | Type | Description |
|----------|------|-------------|
| `BatchDataProvider` | Protocol | `get_fundamental_series(symbol, start, end) → pd.Series` — returns full series at once. Best for file-based data. |
| `PointDataProvider` | Protocol | `get_fundamental_at(symbol, timestamp) → int` — returns single value per call. Best for DB queries, generative models. |
| `DataFrameProvider` | Concrete | Reference `BatchDataProvider` wrapping `Dict[str, pd.Series]`. Slices by timestamp range. |

**Gap**: No built-in CSV or Parquet provider. Users must write their own loader. A `CsvProvider` and `ParquetProvider` would cover the two most common external data formats.

### 5.7 Oracle System Pain Points

1. **Information structure is correct but incomplete**: The informed/uninformed split (ValueAgent → oracle, everyone else → LOB) correctly models real-market information asymmetry. The gap is not that existing agents lack oracle access — it's that the only *type* of informed trading modeled is continuous Bayesian estimation (ValueAgent). There is no agent that reacts to **discrete information events** (megashocks, earnings, news). An `InformedTraderAgent` that receives oracle event signals and trades aggressively would model *adverse selection* — the single most important microstructure phenomenon. When an informed trader hits the book, AMM spreads should widen, liquidity should dry up, and stop-losses should cascade.

2. **No oracle event subscription API**: The megashock system in `SparseMeanRevertingOracle` generates realistic price discontinuities, but they are **invisible to agents**. No agent can detect "a shock just happened." An event subscription API is the key architectural enabler for informed-trader strategies, news-reactive agents, and adversarial stress testing (flash crash simulation, adverse selection spikes).

3. **ExternalDataOracle not compilable**: The `NotImplementedError` at `compiler.py:213` blocks the most production-relevant oracle from declarative configuration. This forces users into manual `compile()` + injection workflows.

---

## 6. Oracle Recommendations

### 6.1 Immediate Fixes

| Fix | File | Detail |
|-----|------|--------|
| Implement `ExternalDataOracleConfig` compilation | `compiler.py:212-217` | Load CSV/Parquet from `data_path`, construct `DataFrameProvider`, pass to `ExternalDataOracle`. |

### 6.2 Strategic Enhancements

| Enhancement | Priority | Detail |
|-------------|:--------:|--------|
| Oracle event subscription API | P1 | Allow agents to register for discrete information shocks (megashocks, earnings). The oracle emits `OracleEventMsg(symbol, event_type, magnitude)` to subscribed agents with per-subscriber delay/noise (natural information asymmetry model). Enables the `InformedTraderAgent` pattern described in §4.1 and unlocks adversarial stress testing scenarios. Delivery via kernel-routed messages (consistent with ABIDES event model) adds natural latency modeling. |
| `ValueAgentConfig` auto-derive `r_bar` | P1 | `_prepare_constructor_kwargs()` should extract `r_bar` from the oracle config, eliminating parameter duplication and misalignment risk. |
| Optional AMM fundamental anchor | P2 | Some sophisticated real-world MMs (Citadel, Virtu) incorporate fundamental views to reduce inventory risk. An optional `use_oracle: bool = False` on `AdaptiveMarketMakerAgent` that blends `alpha * oracle_obs + (1-alpha) * lob_mid` would add realism for specific scenarios. Default off — the current LOB-mid quoting is correct for most MMs. |
| Multi-symbol correlation | P2 | Current oracles generate independent series per symbol. Add covariance structure (e.g., Cholesky-decomposed correlated OU processes) for realistic cross-asset modeling. |
| Built-in CSV/Parquet providers | P2 | Ship `CsvProvider` and `ParquetProvider` alongside `DataFrameProvider` to cover the most common external data ingestion patterns without user boilerplate. |

### 6.3 What NOT to Do

The ratio of informed-to-uninformed agents is a key calibration parameter for realistic microstructure. The following should **not** receive oracle access:

- **MomentumAgent**: Momentum strategies are purely price-based. Oracle access would destroy the informed/uninformed heterogeneity that creates realistic price discovery.
- **Execution agents** (POV, future TWAP/VWAP): These target benchmarks, not alpha. They should not have fundamental views.
- **NoiseAgent**: Axiomatically uninformed.
- **Future LOB-only agents** (StopLoss, Iceberg, MeanReversion): These react to price/book state, not fundamentals. Oracle access would misrepresent their real-world behaviour.

---

## 7. Product Evaluation

### 7.1 Quality Assessment (as of v2.1.0)

All 15 bugs identified in v2.0.0 have been resolved. The cancel-and-repost performance bottleneck was eliminated in v2.1.0 via `replace_order()`. The full 312-test suite passes with zero warnings.

**Remaining technical debt** is exclusively design-level (unbounded positions, hard-coded parameters, missing agent types) — no runtime crashes, data-corruption risks, or performance anti-patterns.

### 7.2 Fitness for Use Case 1 — Simulation Dashboard

> *A dashboard that allows graphical selection, configuration and execution of agent populations, with CSV/database data injection and per-agent metrics.*

| Dimension | Rating | Detail |
|-----------|:------:|--------|
| **Agent Variety** | ⚠️ Weak | Only 5 concrete trading agents. A dashboard user expects to drag-and-drop from a palette of 15–20 agent archetypes (noise, value, momentum, mean-reversion, TWAP, VWAP, POV, MM, stop-loss, iceberg, informed trader, pairs arb, etc.). The current roster covers academic microstructure but not institutional simulation. |
| **Configurability** | ✅ Good | The `SimulationBuilder` + `@register_agent` system is well-designed for a dashboard front-end. Pydantic models provide schema introspection (`get_config_schema()`), enabling auto-generated forms. YAML/JSON serialization supports save/load. |
| **Data Injection** | ⚠️ Partial | `ExternalDataOracle` with `BatchDataProvider`/`PointDataProvider` is architecturally solid, but: (1) `ExternalDataOracleConfig` compilation is `NotImplementedError` — cannot configure via the dashboard's config layer, (2) no built-in `CsvProvider` or `ParquetProvider` — dashboard must ship its own loader. |
| **Metrics & Observability** | ⚠️ Partial | `ExchangeAgent.MetricTracker` captures spread, volume, and book depth. `parse_logs_df` extracts event logs. But: (1) no per-agent P&L tracking except end-of-day `mark_to_market`, (2) no mid-simulation metric queries (snapshot-only at end), (3) `AdaptiveMarketMakerAgent` has no inventory or P&L metrics. For a dashboard, users expect real-time-style metric streams. |
| **Realism** | ⚠️ Moderate | `NoiseAgent` is single-shot (unrealistic). `MomentumAgent` has no position limits (accumulates unbounded inventory). No stop-loss triggered volume. No iceberg orders. The resulting microstructure is recognizably market-like but lacks several stylized facts (e.g., volatility clustering from stop cascades, hidden liquidity). |
| **Reproducibility** | ✅ Good | Hierarchical `RandomState` seeding ensures identical results across runs with same config. `SimulationConfig` is immutable and reusable. |

**Key gaps for dashboard launch:**
1. Ship TWAP + VWAP + StopLoss + MeanReversion agents (minimum viable palette — all LOB-only, no oracle access needed)
2. Implement `ExternalDataOracleConfig` compilation (unblock CSV/DB data injection via config)
3. Add continuous-trading mode to `NoiseAgent` (multi-wake with configurable frequency)
4. Add per-agent P&L / execution quality metrics accessible from `SimulationResult`
5. Ship `InformedTraderAgent` + oracle event subscription API (unlocks adverse selection scenarios; the only new agent that requires oracle access)

### 7.3 Fitness for Use Case 2 — Agentic Adversarial Stress Testing

> *An AI framework reads a strategy, designs adversarial scenarios, runs them via ABIDES, then uses perfect observability to explore results and provide insights.*

| Dimension | Rating | Detail |
|-----------|:------:|--------|
| **Perfect Observability** | ✅ Strong | ABIDES is fully deterministic and single-threaded. Every message, order, fill, and state transition is logged via `logEvent`. `parse_logs_df` reconstructs the full event stream. The AI agent can replay any simulation tick-by-tick. |
| **Scenario Design API** | ✅ Good | `SimulationBuilder` supports programmatic construction of arbitrary scenarios. Templates (`rmsc04`, `liquid_market`, `thin_market`) provide starting points. An AI agent can mutate configs (agent counts, parameters, oracle shocks) and re-run. |
| **Execution Introspection** | ⚠️ Partial | Order-level tracking exists (`self.orders` in `TradingAgent`), and POV now tracks fill prices. But: (1) no standardized execution-quality metrics (VWAP slippage, implementation shortfall, participation rate) are computed automatically, (2) L1/L2 book history requires post-hoc reconstruction from `OrderBook.history`, (3) no causal attribution — the AI cannot easily determine *why* a price moved (which agent's order caused the fill). |
| **Stress Scenario Richness** | ⚠️ Weak | Without stop-loss agents, informed traders, or correlated multi-asset oracles, the adversarial scenario space is limited. The AI can vary agent counts, noise levels, and oracle parameters, but cannot simulate: flash crashes (no stop cascades), adverse selection spikes (no informed trader), or cross-asset contagion (no correlated oracles). |
| **Batch & Parallel Execution** | ✅ Good | `run_batch(configs)` supports parallel multi-simulation runs. An AI agent can sweep parameter spaces efficiently. |
| **AI Discoverability** | ✅ Good | `list_agent_types()`, `get_config_schema()`, `validate_config()` provide programmatic introspection. Config schemas are JSON-serializable — an LLM can parse and generate them. |

**Key gaps for stress testing:**
1. Ship `InformedTraderAgent` + oracle event subscription API (unlock adverse selection scenarios — the key missing information structure; per-subscriber delay/noise enables natural information asymmetry calibration)
2. Ship `StopLossAgent` (unlock cascade/flash-crash scenarios — LOB-only, no oracle needed)
3. Add standardized execution-quality metrics to `SimulationResult` (VWAP slippage, IS, fill rate)
4. Add causal order attribution: tag each trade with the aggressor agent ID and the passive agent ID
5. Add correlated multi-symbol oracle support (unlock cross-asset stress scenarios)

### 7.4 Overall Product Maturity

| Area | Maturity | Comment |
|------|:--------:|---------|
| Core simulation engine | **Production** | Kernel, message passing, order book, matching — well-tested and performant after v1.2.0 optimizations. |
| Configuration system | **Production** | `SimulationBuilder`, registry, compiler, YAML/JSON — comprehensive and AI-friendly. |
| Agent roster | **Alpha** | 5 concrete agents is insufficient for either use case. Minimum viable product needs 10–12 agent types. |
| Oracle system | **Beta** | `SparseMeanRevertingOracle` is solid; informed/uninformed agent split correctly models real-market information asymmetry. `ExternalDataOracle` is architecturally good but not wired into the config system. Missing: event subscription API for discrete information shocks (key enabler for `InformedTraderAgent`). |
| Data extraction | **Beta** | `parse_logs_df` works but requires post-hoc reconstruction. No streaming metrics. |
| Documentation | **Good** | Config system docs, custom agent guide, LLM gotchas, data extraction — all current and accurate. |

**Bottom line**: ABIDES v2.1.0 is a **solid simulation engine with a capable config system** but an **underdeveloped agent ecosystem**. The engine is ready for production; the agent library needs 2–3 more development cycles to support the target use cases.

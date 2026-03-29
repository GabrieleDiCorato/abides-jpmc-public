# ABIDES — Roadmap & Priority List

**Date**: 2026-03-28
**Baseline**: v2.2.0 — all known bugs resolved, 491 tests passing, zero runtime/data-corruption issues.

---

## 1. Current State Summary

### What Works

| Area | Status | Notes |
|------|:------:|-------|
| Simulation engine | **Production** | Kernel, message passing, order book, matching — tested and performant. |
| Config system | **Production** | `SimulationBuilder`, `@register_agent`, compiler, YAML/JSON — AI-friendly with schema introspection. |
| Risk controls | **Production** | `RiskConfig` wires position limits, circuit breakers, per-fill P&L (`FILL_PNL`) through all 5 concrete agents. |
| Oracle system | **Production** | Required oracle field, oracle-absent mode, `ExternalDataOracle` with injection pattern, ValueAgent auto-inherits params. |
| Data extraction | **Good** | `parse_logs_df`, L1/L2/L3 snapshots, `SimulationResult` with `summary_dict()` / `narrative()`, VWAP computation. |
| Parallel execution | **Good** | `run_batch()` with unique log dirs, deterministic seed hierarchy. |
| AI discoverability | **Good** | `list_agent_types()`, `get_config_schema()`, `validate_config()` — designed for LLM tool-calling. |
| Documentation | **Good** | Config system, custom agent guide, LLM gotchas, data extraction — current and accurate. |

### Agent Inventory (5 trading + 1 infrastructure)

| Agent | Category | Registered | Oracle? |
|-------|----------|:----------:|:-------:|
| ExchangeAgent | Infrastructure | N/A | Seeds `last_trade` |
| NoiseAgent | Background | `"noise"` | No |
| ValueAgent | Background | `"value"` | Yes — Bayesian |
| MomentumAgent | Strategy | `"momentum"` | No |
| AdaptiveMarketMakerAgent | Market Maker | `"adaptive_market_maker"` | No |
| POVExecutionAgent | Execution | `"pov_execution"` | No |

The informed/uninformed split is **correct by design** — ValueAgent pushes prices toward fundamentals; everyone else reacts to LOB state. This heterogeneity IS the price discovery mechanism.

### Templates (3 base + 2 overlay)

| Template | Profile |
|----------|---------|
| `rmsc04` | Reference config: 2 MMs, 102 Value, 12 Momentum, 1000 Noise |
| `liquid_market` | High liquidity: 4 MMs, 200 Value, 25 Momentum, 5000 Noise |
| `thin_market` | Low liquidity: no MMs, 20 Value, 100 Noise |
| `with_momentum` | Overlay: adds 12 Momentum agents |
| `with_execution` | Overlay: adds 1 POV Execution agent |

---

## 2. Known Issues (Existing Agents)

| Agent | Issue | Impact |
|-------|-------|--------|
| **NoiseAgent** | Single-shot — places one order then goes silent | Unrealistic microstructure; fewer background messages than real markets |
| **MomentumAgent** | No exit/reversal logic — rides trend indefinitely | Overstates trend-following impact; no stop-loss or profit-target |
| **AdaptiveMarketMakerAgent** | No end-of-day inventory flatten | MM carries residual position through close — unrealistic P&L |
| **POVExecutionAgent** | Market orders only; no urgency parameter | No price improvement; can't model time-adaptive execution |
| **TradingAgent** | Drawdown measured from `starting_cash`, not peak NAV | `_peak_nav` is tracked but not used for enforcement |
| **TradingAgent** | `FILL_PNL` logs NAV but not per-symbol position | Limits per-instrument analysis |

---

## 3. Missing Capabilities

### 3.1 New Agent Types

| Agent | Description | Oracle? | Depends On |
|-------|-------------|:-------:|------------|
| **MeanReversionAgent** | Bollinger-band/z-score contrarian strategy. Complement to MomentumAgent. | No | Nothing — LOB-only |
| **TWAPExecutionAgent** | Uniform time-sliced child orders. Configurable start/end, randomization, limit-vs-market. | No | Nothing |
| **VWAPExecutionAgent** | Volume-profile-weighted execution. Structurally different from POV (benchmark vs. participation rate). | No | Nothing |
| **InformedTraderAgent** | Reacts to discrete oracle events (megashocks, earnings) with configurable delay/precision/aggressiveness. Models adverse selection. | Yes — events | Oracle event subscription API |
| **ImplementationShortfallAgent** | Almgren-Chriss optimal execution balancing market impact vs. timing risk. | No | Nothing |
| **PairsArbitrageAgent** | Trades two correlated symbols on spread z-score. First multi-symbol agent. | No | Correlated oracle |
| **HFTAgent** | Exploits queue priority and latency; cancels stale quotes on price change. | No | Nothing |

### 3.2 Exchange Features

Iceberg and stop orders are **exchange-level mechanisms** — implementing them as agents introduces unrealistic latency. The matching engine must manage them.

| Feature | Detail |
|---------|--------|
| **Time-in-force (IOC, FOK, DAY)** | All orders are currently GTC. IOC is essential for execution algos that avoid resting orders. Simplest exchange change. |
| **Stop orders** | No `stop_price` / triggered-order queue. Exchange should convert stops to market/limit atomically on trigger. Enables flash-crash / cascade scenarios. |
| **Iceberg / reserve orders** | `PriceLevel` already separates visible/hidden queues. Add `display_quantity` to `LimitOrder`; auto-refresh visible slice on fill. |
| **Exchange-level circuit breakers** | No trading halts or LULD bands. Only agent-level via `RiskConfig`. Shapes tail-risk dynamics. |
| **Self-trade prevention** | No same-agent matching check. Real exchanges reject or cancel self-trades. |
| **Opening / closing auction** | No call-auction. Open seeds from oracle, not a crossing mechanism. Auctions handle ~15-20% of daily volume. |

### 3.3 Oracle Enhancements

| Enhancement | Detail |
|-------------|--------|
| **Oracle event subscription API** | Agents register for discrete shocks; oracle emits `OracleEventMsg` via kernel-routed messages with per-subscriber delay/noise. Key enabler for `InformedTraderAgent`. |
| **Correlated multi-symbol oracle** | Each symbol's OU process is currently independent. Cholesky-decomposed correlated processes needed for cross-asset scenarios. |
| **Built-in CSV/Parquet providers** | Only `DataFrameProvider` exists. `CsvProvider` and `ParquetProvider` would eliminate external boilerplate. |
| **Optional AMM fundamental anchor** | Blend `alpha * oracle + (1-alpha) * LOB mid` for sophisticated MM modeling. Default off. |

### 3.4 Observability & Metrics

| Issue | Detail |
|-------|--------|
| **No standardized execution-quality metrics** | VWAP slippage, implementation shortfall, participation rate not auto-computed. |
| **No causal order attribution** | Cannot determine which agent caused a price move. Tag trades with aggressor/passive agent ID. |
| **FILL_PNL aggregation** | Per-fill events exist but aren't aggregated into `SimulationResult`. |

### 3.5 Existing Agent Improvements

| Issue | Detail |
|-------|--------|
| **NoiseAgent multi-wake mode** | Configurable continuous trading (wake frequency parameter) instead of single-shot. |
| **MomentumAgent exit logic** | Configurable trailing stop / profit target / reversal signal. |
| **AMM end-of-day flatten** | Schedule market orders to flatten inventory N minutes before close. |
| **POV limit-order mode** | Support limit orders with active crossing for price improvement. |
| **Peak-drawdown enforcement** | Use `_peak_nav` (already tracked) as drawdown reference instead of `starting_cash`. |

### 3.6 Housekeeping

| Issue | Detail |
|-------|--------|
| **Register gym agents** | `CoreBackgroundAgent` and `FinancialGymAgent` not in config system. |
| **Remove MeanRevertingOracleConfig** | Deprecated oracle is still compilable via config. Remove config model to eliminate foot-gun. |

---

## 4. Priority List

Items are ordered by combined value: **ABIDES as a library** × **agentic stress-testing use case**. Dependencies are noted — some items unlock others.

### P0 — High value for both ABIDES and stress-testing

| # | Item | Type | Status | Rationale |
|---|------|:----:|:------:|-----------|
| 1 | **MeanReversionAgent** | New agent | ✅ Done | Cheapest new agent (follows MomentumAgent pattern). Adds contrarian flow — critically important for adversarial scenarios ("what if contrarians fight your momentum strategy?"). No dependencies. |
| 2 | **NoiseAgent multi-wake mode** | Agent fix | ✅ Done | Single-shot NoiseAgent produces unrealistically sparse background flow. Continuous noise is essential for meaningful microstructure in any scenario. Small change, large impact on simulation quality. |
| 3 | **Time-in-force (IOC, FOK, DAY)** | Exchange | ✅ Done | Simplest exchange enhancement. IOC is prerequisite for realistic execution agents (TWAP/VWAP). DAY orders eliminate stale-order cleanup hacks. Required for any new execution agent to be credible. |
| 4 | **Execution-quality metrics in SimulationResult** | Observability | ✅ Done | VWAP slippage, fill rate, participation rate. Without these, an AI agent has no structured way to grade strategy performance. Critical for stress-testing interpretation. Builds on existing `summary_dict()`. |

### P1 — High value for ABIDES; enables next tier of stress scenarios

| # | Item | Type | Rationale |
|---|------|:----:|-----------|
| 5 | **TWAPExecutionAgent** | New agent | Most basic institutional algo. Follows POV state-machine pattern. Requires IOC for non-resting slices. Adds institutional background flow to scenarios. |
| 6 | **VWAPExecutionAgent** | New agent | Industry-standard execution benchmark. Structurally different from POV (price benchmark vs. participation). Build alongside TWAP — shared execution infrastructure. |
| 7 | **Stop orders (exchange-level)** | Exchange | Enables flash-crash / cascade scenarios — stop-triggered volume drives volatility clustering. Agent-side polling distorts timing. Unlocks a major class of stress-testing scenarios. |
| 8 | **AMM end-of-day flatten** | Agent fix | Affects P&L realism for every simulation with market makers. Small targeted change in `AdaptiveMarketMakerAgent`. |
| 9 | **Causal order attribution** | Observability | Tag trades with aggressor/passive agent ID. Lets the AI agent explain *why* prices moved — essential for actionable stress-test feedback. |
| 10 | **FILL_PNL aggregation in SimulationResult** | Observability | Surface per-agent equity curves without post-hoc log parsing. Makes `summary_dict()` self-contained. |

### P2 — Significant value; larger effort or narrower use case

| # | Item | Type | Rationale |
|---|------|:----:|-----------|
| 11 | **Oracle event subscription API** | Architecture | Prerequisite for InformedTraderAgent. Exposes megashocks to agents via kernel messages with configurable delay/noise. Significant architecture work but unlocks adverse-selection scenarios. |
| 12 | **InformedTraderAgent** | New agent | Reacts to discrete oracle events. Models adverse selection — the key missing microstructure phenomenon. Depends on #11. |
| 13 | **Iceberg orders (exchange-level)** | Exchange | `PriceLevel` already separates visible/hidden — natural extension. Adds hidden liquidity dynamics. |
| 14 | **MomentumAgent exit logic** | Agent fix | Trailing stop / profit target. Improves realism of trend-following flow. |
| 15 | **POV limit-order mode + urgency** | Agent fix | Support limit orders and time-adaptive participation. |
| 16 | **Peak-drawdown enforcement** | Risk | Use already-tracked `_peak_nav` as drawdown reference. Small change in `TradingAgent`. |
| 17 | **Per-symbol position in FILL_PNL** | Observability | Add `holdings[symbol]` to FILL_PNL payload. |
| 18 | **Built-in CSV/Parquet providers** | Data | Convenience for external data injection. `DataFrameProvider` exists as workaround. |

### P3 — Differentiation / future

| # | Item | Type | Rationale |
|---|------|:----:|-----------|
| 19 | **ImplementationShortfallAgent** | New agent | Almgren-Chriss optimal execution. Academic interest; specialized use case. |
| 20 | **Exchange-level circuit breakers (LULD)** | Exchange | Trading halts on extreme moves. Important for tail-risk but modeled at agent level today. |
| 21 | **Self-trade prevention** | Exchange | Correctness improvement. Agents work around it today. |
| 22 | **Correlated multi-symbol oracle** | Oracle | Cholesky-decomposed correlated OU processes. Required for PairsArbitrageAgent and cross-asset stress. |
| 23 | **PairsArbitrageAgent** | New agent | First multi-symbol agent. Depends on #22. |
| 24 | **HFTAgent** | New agent | Queue priority exploitation. Niche but differentiating. |
| 25 | **Optional AMM fundamental anchor** | Agent enhancement | Blend oracle + LOB mid. Default off. Niche. |
| 26 | **Opening / closing auction** | Exchange | Call-auction matching. Important for realism but significant effort. |
| 27 | **Register gym agents** | Housekeeping | `CoreBackgroundAgent`, `FinancialGymAgent` not in config system. |
| 28 | **Remove MeanRevertingOracleConfig** | Housekeeping | Deprecated oracle still compilable. Remove config to eliminate foot-gun. |

---

## 5. Design Constraints

### Oracle Access Rules

Only agents that model **informed** trading should access the oracle. This preserves the informed/uninformed heterogeneity that drives realistic price discovery.

| Should access oracle | Should NOT |
|---------------------|------------|
| ValueAgent (Bayesian estimation) | MomentumAgent (price-based by definition) |
| Future InformedTraderAgent (event-driven) | Execution agents — POV, TWAP, VWAP (benchmark, not alpha) |
| | NoiseAgent (axiomatically uninformed) |
| | MeanReversionAgent (LOB-only) |

### Exchange vs. Agent Responsibility

Stop orders, iceberg orders, and circuit breakers belong in the **matching engine**, not in agent code. Agent-side implementations introduce inter-message latency that distorts the phenomena they're meant to model (stop cascades, hidden liquidity refresh, trading halts).

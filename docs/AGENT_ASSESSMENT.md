# ABIDES — Roadmap & Priority List

**Date**: 2026-03-30
**Baseline**: v2.2.0 — all P0/P1 work complete, 979 tests passing, zero runtime/data-corruption issues.

---

## 1. Current State Summary

### What Works

| Area | Status | Notes |
|------|:------:|-------|
| Simulation engine | **Production** | Kernel, message passing, order book, matching — tested and performant. |
| Config system | **Production** | `SimulationBuilder`, `@register_agent`, compiler, YAML/JSON — AI-friendly with schema introspection. |
| Risk controls | **Production** | `RiskConfig` wires position limits, circuit breakers, per-fill P&L (`FILL_PNL`) through all 8 concrete agents. |
| Oracle system | **Production** | Required oracle field, oracle-absent mode, `ExternalDataOracle` with injection pattern, ValueAgent auto-inherits params. |
| Order types | **Production** | `TimeInForce` (GTC, IOC, FOK, DAY), `StopOrder` with exchange-side trigger queue. |
| Observability | **Production** | `ExecutionMetrics`, `TradeAttribution`, `EquityCurve` in `SimulationResult`. Profile-gated extraction. |
| Data extraction | **Good** | `parse_logs_df`, L1/L2/L3 snapshots, `SimulationResult` with `summary_dict()` / `narrative()`, VWAP computation. |
| Parallel execution | **Good** | `run_batch()` with unique log dirs, deterministic seed hierarchy. |
| AI discoverability | **Good** | `list_agent_types()`, `get_config_schema()`, `validate_config()` — designed for LLM tool-calling. |
| Documentation | **Good** | Config system, custom agent guide, LLM gotchas, data extraction — current and accurate. |

### Agent Inventory (8 trading + 1 infrastructure)

| Agent | Category | Registered | Oracle? |
|-------|----------|:----------:|:-------:|
| ExchangeAgent | Infrastructure | N/A | Seeds `last_trade` |
| NoiseAgent | Background | `"noise"` | No |
| ValueAgent | Background | `"value"` | Yes — Bayesian |
| MomentumAgent | Strategy | `"momentum"` | No |
| MeanReversionAgent | Strategy | `"mean_reversion"` | No |
| AdaptiveMarketMakerAgent | Market Maker | `"adaptive_market_maker"` | No |
| POVExecutionAgent | Execution | `"pov_execution"` | No |
| TWAPExecutionAgent | Execution | `"twap_execution"` | No |
| VWAPExecutionAgent | Execution | `"vwap_execution"` | No |

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
| **MomentumAgent** | No exit/reversal logic — rides trend indefinitely | Overstates trend-following impact; no stop-loss or profit-target |
| **POVExecutionAgent** | Market orders only; no urgency parameter | No price improvement; can't model time-adaptive execution |
| **TradingAgent** | Drawdown measured from `starting_cash`, not peak NAV | `_peak_nav` is tracked but not used for enforcement |
| **TradingAgent** | `FILL_PNL` logs NAV but not per-symbol position | Limits per-instrument analysis |

---

## 3. Missing Capabilities

### 3.1 New Agent Types

| Agent | Description | Oracle? | Depends On |
|-------|-------------|:-------:|------------|
| **InformedTraderAgent** | Reacts to discrete oracle events (megashocks, earnings) with configurable delay/precision/aggressiveness. Models adverse selection. | Yes — events | Oracle event subscription API |
| **ImplementationShortfallAgent** | Almgren-Chriss optimal execution balancing market impact vs. timing risk. | No | Nothing |
| **PairsArbitrageAgent** | Trades two correlated symbols on spread z-score. First multi-symbol agent. | No | Correlated oracle |
| **HFTAgent** | Exploits queue priority and latency; cancels stale quotes on price change. | No | Nothing |

### 3.2 Exchange Features

Iceberg orders are **exchange-level mechanisms** — implementing them as agents introduces unrealistic latency. The matching engine must manage them.

| Feature | Detail |
|---------|--------|
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

### 3.4 Existing Agent Improvements

| Issue | Detail |
|-------|--------|
| **MomentumAgent exit logic** | Configurable trailing stop / profit target / reversal signal. |
| **POV limit-order mode** | Support limit orders with active crossing for price improvement. |
| **Peak-drawdown enforcement** | Use `_peak_nav` (already tracked) as drawdown reference instead of `starting_cash`. |

---

## 4. Priority List

Items are ordered by combined value: **ABIDES as a library** × **agentic stress-testing use case**. Dependencies are noted — some items unlock others.

### P0 — High value; larger effort or narrower use case

| # | Item | Type | Rationale |
|---|------|:----:|-----------|
| 1 | **Oracle event subscription API** | Architecture | Prerequisite for InformedTraderAgent. Exposes megashocks to agents via kernel messages with configurable delay/noise. Significant architecture work but unlocks adverse-selection scenarios. |
| 2 | **InformedTraderAgent** | New agent | Reacts to discrete oracle events. Models adverse selection — the key missing microstructure phenomenon. Depends on #1. |
| 3 | **Iceberg orders (exchange-level)** | Exchange | `PriceLevel` already separates visible/hidden — natural extension. Adds hidden liquidity dynamics. |
| 4 | **MomentumAgent exit logic** | Agent fix | Trailing stop / profit target. Improves realism of trend-following flow. |
| 5 | **POV limit-order mode + urgency** | Agent fix | Support limit orders and time-adaptive participation. |
| 6 | **Peak-drawdown enforcement** | Risk | Use already-tracked `_peak_nav` as drawdown reference. Small change in `TradingAgent`. |
| 7 | **Per-symbol position in FILL_PNL** | Observability | Add `holdings[symbol]` to FILL_PNL payload. |
| 8 | **Built-in CSV/Parquet providers** | Data | Convenience for external data injection. `DataFrameProvider` exists as workaround. |

### P1 — Differentiation / future

| # | Item | Type | Rationale |
|---|------|:----:|-----------|
| 9 | **ImplementationShortfallAgent** | New agent | Almgren-Chriss optimal execution. Academic interest; specialized use case. |
| 10 | **Exchange-level circuit breakers (LULD)** | Exchange | Trading halts on extreme moves. Important for tail-risk but modeled at agent level today. |
| 11 | **Self-trade prevention** | Exchange | Correctness improvement. Agents work around it today. |
| 12 | **Correlated multi-symbol oracle** | Oracle | Cholesky-decomposed correlated OU processes. Required for PairsArbitrageAgent and cross-asset stress. |
| 13 | **PairsArbitrageAgent** | New agent | First multi-symbol agent. Depends on #12. |
| 14 | **HFTAgent** | New agent | Queue priority exploitation. Niche but differentiating. |
| 15 | **Optional AMM fundamental anchor** | Agent enhancement | Blend oracle + LOB mid. Default off. Niche. |
| 16 | **Opening / closing auction** | Exchange | Call-auction matching. Important for realism but significant effort. |
| 17 | **Register gym agents** | Housekeeping | `CoreBackgroundAgent`, `FinancialGymAgent` not in config system. |
| 18 | **Remove MeanRevertingOracleConfig** | Housekeeping | Deprecated oracle still compilable. Remove config to eliminate foot-gun. |

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

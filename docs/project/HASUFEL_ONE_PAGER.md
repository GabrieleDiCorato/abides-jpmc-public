# ABIDES-Hasufel — Research-Grade Market Microstructure Simulation for Strategy Research

---

## The Problem

Financial institutions need to test trading strategies, execution algorithms, and risk controls *before* deploying real capital. The available options are expensive, incomplete, or both:

- **Historical backtesting** replays past data but cannot answer "what if" questions — it assumes the strategy does not move the market.
- **Proprietary simulators** (e.g. Goldman's SecDB, JPMC's Athena) are locked behind institutional walls.
- **Academic agent-based models** exist but typically lack production-grade execution logic, risk controls, and structured analytics — leaving a gap between research insight and deployable prototype.

ABIDES (Byrd & Balch, 2019) was a promising academic simulator from Georgia Tech, later maintained by J.P. Morgan Chase, but its public release was archived with unresolved runtime bugs, no configuration system, and no analytics pipeline.

## The Contribution: ABIDES-Hasufel

ABIDES-Hasufel **revives, modernises, and substantially extends** the ABIDES simulator into a research-ready platform where AI-driven strategies can be developed, tested, and measured under realistic market conditions — with full transparency and reproducibility.

### What Makes It Novel

| Capability | State of the Art | ABIDES-Hasufel |
|:---|:---|:---|
| **Declarative configuration** | Procedural scripts; hard to reproduce | YAML/JSON configs, fluent `SimulationBuilder`, immutable `SimulationConfig` — same config always yields same result |
| **Composition-invariant RNG** | Sequential seed draws — adding an agent shifts all other agents' behaviour | SHA-256 identity hashing — inject a new strategy and *all other agents behave identically* to the baseline |
| **Rich agent taxonomy** | Noise + value + 1 market maker | 9 agent types across 4 categories: background, strategy, execution (TWAP/VWAP/POV), market making |
| **Institutional risk controls** | Not present in academic simulators | Per-agent position limits, circuit breakers, drawdown kill-switch, order-rate limiters — all declarative via `RiskConfig` |
| **Order type coverage** | Limit and market only | + IOC, FOK, DAY time-in-force qualifiers, stop orders with exchange-side trigger |
| **Quantitative analytics** | Manual log parsing | `compute_rich_metrics()` → Sharpe, max drawdown, VWAP, LOB imbalance, resilience, VPIN, OTT ratio, adverse selection — all as typed Pydantic models |
| **Parallel experimentation** | Manual scripting | `run_batch()` with multiprocessing, deterministic per-worker seeding, UUID-isolated logs |
| **AI/LLM integration** | None | Schema-introspectable configs, `@register_agent` plugin system, structured `SimulationResult` for tool-calling agents |

### Key Features in Detail

**1. Discrete-Event Market Simulation Engine** — A NASDAQ-like exchange with a continuous double-auction order book, nanosecond-resolution event scheduling, and configurable network latency. Prices are integer cents everywhere, matching real exchange precision.

**2. Pluggable Agent Architecture** — Heterogeneous agent populations drive emergent price discovery: noise traders provide liquidity, value agents form Bayesian estimates of fundamental value, momentum agents exploit trends, and market makers quote two-sided depth. New strategies plug in via `@register_agent` with zero framework changes.

**3. Execution Algorithm Suite** — TWAP, VWAP, and POV execution agents slice parent orders across the trading session. Each inherits from a shared `BaseSlicingExecutionAgent` with fill tracking, arrival-price capture, and IOC child order submission — mirroring how institutional execution desks operate.

**4. Risk Management Layer** — Every trading agent inherits `RiskConfig`: symmetric position limits with clamp-or-block enforcement, maximum drawdown kill-switch (latching), and tumbling-window order-rate throttling. Risk is checked on every order submission and every fill.

**5. Quantitative Metrics Pipeline** — A single `compute_rich_metrics()` call produces agent-level analytics (PnL, Sharpe, drawdown, fill rate, inventory volatility) and market microstructure indicators (LOB imbalance per Cont, Kukanov & Stoikov 2014; resilience per Foucault, Kadan & Kandel 2013; VPIN per Easley et al. 2012; OTT ratio per MiFID II RTS 9). All metrics degrade gracefully when data is unavailable.

**6. Scenario Templates** — Five pre-built market regimes (`stable_day`, `volatile_day`, `low_liquidity`, `trending_day`, `stress_test`) plus overlay templates for controlled experimentation. Each template is composable and fully reproducible from a single seed integer.

### Simulation Output: Before and After

The original ABIDES produced a single opaque `end_state` dictionary. Extracting any useful information required manual log parsing, DataFrame archaeology, and intimate knowledge of agent internals. There was no structured result, no typed analytics, and no way to compare runs programmatically.

ABIDES-Hasufel replaces this with a **layered, profile-gated output system** — from a lightweight summary to full per-order lifecycle tracking — all returned as immutable, typed Pydantic models.

| What you get | Original ABIDES | ABIDES-Hasufel |
|:---|:---|:---|
| **Simulation result** | Raw dict; parse it yourself | `SimulationResult` — immutable, typed, with `.summary()` and `.summary_dict()` |
| **Per-agent PnL** | Manually compute from `holdings` dict and `last_trade` | `AgentData.pnl_cents`, `pnl_pct`, `mark_to_market_cents` — mark-to-market, always present |
| **Agent categorisation** | Not tracked | `agent_category` stamped from registry; query with `get_agents_by_category()` |
| **Equity curves** | Not available | `EquityCurve` — per-fill or L1-sampled NAV time-series with high-water mark |
| **Sharpe ratio** | Not available | Annualised from equity curve, with minimum-observation guard |
| **Max drawdown** | Not available | Peak-to-trough in cents; `0` if monotonically increasing |
| **L1 / L2 book history** | Raw order book object; manual extraction | `L1Snapshots`, `L2Snapshots` — NumPy arrays, profile-gated |
| **Trade attribution** | Not tracked | `TradeAttribution` — links every fill to passive and aggressive agent |
| **Execution quality** | Not available | `ExecutionMetrics` — VWAP slippage, participation rate, implementation shortfall, arrival-price comparison |
| **Fill slippage** | Not available | Per-fill signed slippage vs. L1 mid (basis points) |
| **Adverse selection** | Not available | Per-fill mid-price move at configurable look-ahead windows |
| **Order lifecycle** | Not tracked | `OrderLifecycle` — per-order status, resting time, per-fill detail tuples |
| **Fill rate** | Not available | Order-level fill rate (executed / submitted) per agent and market-wide |
| **Microstructure metrics** | Not available | LOB imbalance (Cont et al. 2014), resilience (Foucault et al. 2013), VPIN (Easley et al. 2012), OTT ratio (MiFID II RTS 9) |
| **Volatility** | Not available | Annualised mid-price return std from two-sided L1 observations |
| **Extraction control** | All or nothing | `ResultProfile` flags: `SUMMARY` (KB-scale) → `QUANT` (L1/L2 + analytics) → `FULL` (raw logs) |

**Single-call convenience API:**

```python
rich = compute_rich_metrics(result, include_fills=True,
                            adverse_selection_windows=["100ms", "1s"])
```

Returns `RichSimulationMetrics` with per-agent analytics and per-symbol microstructure indicators in one call. Every field degrades gracefully to `None` when the required data profile is absent — no exceptions, no surprises.

**Standalone compute functions** — all 20+ metric functions (`compute_sharpe_ratio`, `compute_lob_imbalance`, `compute_vpin`, `compute_resilience`, etc.) are exposed as independent, stateless functions that accept plain Python data. No simulation required — feed them your own arrays and get institutional-grade analytics.

### Quality & Engineering

- **1 140 automated tests** covering order book invariants, agent lifecycle, risk enforcement, configuration integrity, and long-simulation health.
- **42 critical bug fixes** from the original ABIDES codebase — including ValueAgent Bayesian update errors, market maker crashes on empty books, and exchange price-recording failures.
- **O(log N) order book operations** (bisect-based), dictionary-dispatch message handling (O(1) vs O(N) isinstance chains), and bounded collections to prevent memory growth.

## Use Cases

- **Strategy Research:** Develop and backtest trading algorithms in a controlled agent-based environment where market impact is endogenous — not assumed away.
- **Execution Quality Analysis:** Measure TWAP/VWAP slippage, adverse selection, and implementation shortfall across market regimes.
- **Risk Model Validation:** Stress-test position limits and circuit breakers before live deployment.
- **Regulatory Analytics:** Compute MiFID II order-to-trade ratios and book quality metrics from simulated order flow.
- **AI/RL Integration:** Use the OpenAI Gym interface (`abides-gym`) to train reinforcement-learning agents in a realistic market environment.

## Technology

Built on Python 3.11+, Pydantic v2, NumPy, and Pandas. Managed with UV for reproducible dependency resolution. CI via GitHub Actions with full test, lint (ruff, black, isort), and type-checking (mypy, pyright) pipelines. BSD-3-Clause licensed.

---

> ABIDES-Hasufel bridges the gap between academic market simulation and institutional-grade strategy research tooling — providing the configurability, analytics, and rigour that quantitative teams require, in an open-source package designed for AI-era workflows.

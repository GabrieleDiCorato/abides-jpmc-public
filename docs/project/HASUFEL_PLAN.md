# ABIDES-Hasufel — Improvement Priority List

**Updated**: 2026-04-22
**Baseline**: v2.5.8 — 1 140 tests, 9 agent types, rich metrics pipeline, composition-invariant RNG.

> **Purpose:** Prioritised list of unimplemented improvements.
> Each item includes motivation, scope, and implementation hints.
> Items are ordered by value; dependencies are noted.
> Review before implementation — some items may be superseded by project direction changes.

---

## Design Constraints

These constraints apply to **all** items below.

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

---

## P0 — High Value

### 1. Oracle Event Subscription API

**Type:** Architecture | **Effort:** Large | **Unlocks:** #2 (InformedTraderAgent)

**Problem:** The oracle generates megashocks and fundamental-value jumps, but agents can only observe prices by polling `observe_price()`. There is no mechanism for agents to *subscribe* to discrete oracle events (earnings releases, regime shifts, shocks) with configurable delay and noise.

**Motivation:** Adverse selection — the primary microstructure phenomenon driving bid-ask spread dynamics — cannot be modeled without informed agents that react to fundamental events before prices adjust. This is the single largest realism gap in ABIDES.

**Implementation hints:**
- Define `OracleEventMsg` as a new message type carrying `(event_type, symbol, magnitude, oracle_time_ns)`.
- Add `subscribe_to_events(agent_id, delay_ns, noise_std)` to the oracle ABC.
- `SparseMeanRevertingOracle` already generates megashocks internally — route them through the kernel message system with per-subscriber delay/noise injection.
- Config: add `oracle_event_delay` and `oracle_event_noise` fields to agent configs that opt in.

### 2. InformedTraderAgent

**Type:** New agent | **Effort:** Medium | **Depends on:** #1

**Problem:** No agent in ABIDES models informed trading — i.e. trading on private information about fundamental value events before the market fully incorporates them. ValueAgent estimates the fundamental continuously; it does not react to discrete shocks.

**Motivation:** Informed traders are the counterparty that market makers fear. Without them, quoted spreads in simulation are narrower than reality, and adverse-selection metrics (`compute_adverse_selection()`) lack a causal driver.

**Implementation hints:**
- Subscribe to oracle events via #1's API.
- On event: submit aggressive limit/market orders proportional to `(event_magnitude × aggressiveness)` with configurable reaction delay.
- Config params: `aggressiveness`, `reaction_delay`, `position_limit`, `symbols`.
- Register as `"informed_trader"` (category: strategy). Oracle-dependent: yes.
- Test: verify that adverse selection bps increase when InformedTraderAgent is present vs. baseline.

### 3. Iceberg / Reserve Orders (Exchange-Level)

**Type:** Exchange | **Effort:** Medium

**Problem:** `PriceLevel` already separates `visible_orders` and `hidden_orders` queues, but there is no `display_quantity` field on `LimitOrder` and no auto-refresh mechanism after partial fills.

**Motivation:** Hidden liquidity is a core feature of modern electronic exchanges. Iceberg orders affect book depth perception, LOB imbalance metrics, and market maker quoting strategy. Without them, simulated books over-represent visible depth.

**Implementation hints:**
- Add `display_quantity: int | None` to `LimitOrder`. When set, `PriceLevel` exposes only `display_quantity` shares; on fill, auto-refresh the visible slice from the hidden reserve.
- Update `get_L1_snapshots()` and `get_L2_snapshots()` to reflect visible-only quantities.
- Add `TradingAgent.place_iceberg_order()` convenience method.
- Config: add `iceberg_display_quantity` to agent configs that need it (primarily market makers).

### 4. MomentumAgent Exit Logic

**Type:** Agent improvement | **Effort:** Small

**Problem:** MomentumAgent enters positions on MA crossover but has no mechanism to exit — it rides trends indefinitely with no stop-loss, profit target, or reversal signal.

**Motivation:** Overstates trend-following market impact in simulations. A realistic momentum agent would have risk-aware exit conditions.

**Implementation hints:**
- Add config fields: `trailing_stop_pct: float | None`, `profit_target_pct: float | None`, `exit_on_reversal: bool`.
- Track entry price per position. On each wakeup, check exit conditions before checking entry signals.
- Register updated params in `MomentumAgentConfig`. Defaults should preserve current behaviour (`None` / `False`).

### 5. POV Limit-Order Mode + Urgency

**Type:** Agent improvement | **Effort:** Small–Medium

**Problem:** `POVExecutionAgent` (and `BaseSlicingExecutionAgent`) uses market orders exclusively. No price improvement is possible, and there is no urgency parameter to adapt participation rate over time.

**Motivation:** Real execution algorithms use limit orders for passive fills and increase urgency as deadlines approach. Market-only POV overstates market impact.

**Implementation hints:**
- Add `order_style: Literal["market", "limit", "adaptive"]` to `BaseSlicingExecutionAgent`.
- `"limit"`: place IOC limit at current best bid/ask. `"adaptive"`: start with limit, switch to market as `urgency` rises toward deadline.
- Add `urgency_curve: Literal["linear", "convex"] | None` for time-varying participation.
- Update `POVExecutionAgentConfig` and `BaseSlicingExecutionAgentConfig`.

### 6. Peak-NAV Drawdown Enforcement

**Type:** Risk | **Effort:** Small

**Problem:** `TradingAgent._check_drawdown()` computes drawdown from `starting_cash`, not from peak NAV. `_peak_nav` is already tracked and updated on every fill, but never used for enforcement.

**Motivation:** Risk managers measure drawdown from the high-water mark, not from initial capital. The current implementation underestimates drawdown for agents that have been profitable before declining.

**Implementation hints:**
- In `TradingAgent._check_drawdown()`, replace `self.starting_cash - mark_to_market()` with `self._peak_nav - mark_to_market()`.
- Update `RiskConfig` docs to clarify "max drawdown from peak NAV."
- Ensure backward compatibility: agents without fills still use `starting_cash` as initial `_peak_nav` (already the case).

### 7. Built-in CSV/Parquet Data Providers

**Type:** Data infrastructure | **Effort:** Small

**Problem:** `ExternalDataOracle` supports a `BatchDataProvider` protocol and ships a `DataFrameProvider`. A minimal `CsvProvider(path)` already exists in `oracles/data_providers.py` as a skeleton, but it has no column-name parameters and no validation. `ParquetProvider` is absent entirely.

**Motivation:** Convenience for researchers who have historical data in CSV or Parquet format and want to replay it through the simulator.

**Implementation hints:**
- Extend `CsvProvider` to accept `symbol_col`, `time_col`, `price_col` keyword arguments. Validate column existence and price-in-cents convention at construction time.
- Add `ParquetProvider(path, symbol_col, time_col, price_col)` with the same interface, backed by `pd.read_parquet()`.
- Both implement `BatchDataProvider` protocol; serve data via `get_data()`.

---

## P1 — Differentiation / Future

### 8. ImplementationShortfallAgent

**Type:** New agent | **Effort:** Medium

**Problem:** No execution agent implements Almgren-Chriss optimal execution, which balances urgency risk (price drift) against market impact (aggressive execution).

**Motivation:** Implementation shortfall is the standard institutional benchmark for execution quality. An agent that optimizes it would complement the existing TWAP/VWAP/POV suite and enable meaningful `implementation_shortfall_bps` metric comparisons.

**Implementation hints:**
- Subclass `BaseSlicingExecutionAgent`. Compute optimal trading trajectory from Almgren-Chriss closed-form solution given `(target_qty, risk_aversion, volatility_estimate, impact_coefficient)`.
- Slice sizes vary over time (front-loaded when risk-averse, back-loaded when impact-averse).
- Register as `"implementation_shortfall"` (category: execution).

### 9. Exchange-Level Circuit Breakers (LULD)

**Type:** Exchange | **Effort:** Medium

**Problem:** No trading halts or Limit-Up/Limit-Down bands exist. Only agent-level circuit breakers via `RiskConfig`.

**Motivation:** Exchange-wide halts shape tail-risk dynamics and affect all agents simultaneously. Important for stress-test realism (flash crash scenarios). MiFID II and Reg NMS both require them.

**Implementation hints:**
- Add `luld_band_pct: float | None` to `ExchangeAgent` config. When set, compute reference price bands at open (or rolling).
- Reject/queue orders outside bands. Trigger a configurable halt duration when consecutive rejections exceed threshold.
- Emit `TradingHaltMsg` to all subscribed agents.

### 10. Self-Trade Prevention

**Type:** Exchange | **Effort:** Small

**Problem:** The matching engine does not check whether the aggressive and passive sides of a match belong to the same agent. Real exchanges reject or cancel self-trades.

**Motivation:** Correctness. Self-trades artificially inflate volume and can distort execution metrics. Agents currently work around this by not quoting aggressively, but the exchange should enforce it.

**Implementation hints:**
- In `OrderBook.execute_trade()`, check `incoming_order.agent_id != resting_order.agent_id`. If equal, skip the match (cancel-newest or cancel-oldest, configurable).
- Add `self_trade_prevention: Literal["cancel_newest", "cancel_oldest", "none"]` to exchange config.

### 11. Correlated Multi-Symbol Oracle

**Type:** Oracle | **Effort:** Medium | **Unlocks:** #12 (PairsArbitrageAgent)

**Problem:** Each symbol's OU process is independent. No correlation structure exists between symbols.

**Motivation:** Cross-asset strategies (pairs trading, statistical arbitrage) require correlated fundamental-value processes. Also needed for portfolio-level stress testing.

**Implementation hints:**
- Accept a correlation matrix in `SparseMeanRevertingOracleConfig`.
- At each time step, draw correlated shocks via Cholesky decomposition of the correlation matrix.
- Fall back to independent processes when no matrix is provided.

### 12. PairsArbitrageAgent

**Type:** New agent | **Effort:** Medium | **Depends on:** #11

**Problem:** No multi-symbol agent exists. All current agents trade a single symbol.

**Motivation:** Pairs trading is the simplest cross-asset strategy and a gateway to more complex multi-instrument scenarios.

**Implementation hints:**
- Track spread z-score between two configured symbols. Enter long/short pair when z > entry threshold; exit when z reverts to exit threshold.
- First agent to subscribe to two exchange agents. Config must accept `symbol_pair: tuple[str, str]`.
- Register as `"pairs_arbitrage"` (category: strategy).

### 13. HFTAgent

**Type:** New agent | **Effort:** Medium

**Problem:** No agent models high-frequency trading behaviour — queue priority exploitation, latency-sensitive quote cancellation, or speed-based adverse selection.

**Motivation:** HFT flow is a significant component of modern market microstructure. Its absence means simulated queue dynamics and fill rates are less realistic.

**Implementation hints:**
- Subscribe to L1 data. On price change: cancel stale quotes immediately, re-quote at new best.
- Exploit latency advantage (configured as lower `computation_delay`).
- Register as `"hft"` (category: market_maker).

### 14. Optional AMM Fundamental Anchor

**Type:** Agent enhancement | **Effort:** Small

**Problem:** `AdaptiveMarketMakerAgent` quotes around the LOB mid-price only. It has no option to blend in fundamental value from the oracle.

**Motivation:** Sophisticated market makers anchor quotes to a fair-value estimate, not just the mid. Blending `alpha × oracle + (1-alpha) × LOB mid` would allow modeling MM information advantage.

**Implementation hints:**
- Add `oracle_anchor_weight: float = 0.0` to `AdaptiveMarketMakerConfig`. At 0.0 (default), behaviour is unchanged.
- When > 0, query oracle `observe_price()` and blend with LOB mid for quote placement.
- Requires oracle access — document this in the oracle access rules.

### 15. Opening / Closing Auction

**Type:** Exchange | **Effort:** Large

**Problem:** Market open seeds price from oracle directly. There is no call-auction mechanism for price discovery at open or close.

**Motivation:** Opening and closing auctions handle 15–20% of daily volume on major exchanges. Their absence means opening price dynamics and EOD concentration effects are not modeled.

**Implementation hints:**
- Add an `AuctionPhase` state to `ExchangeAgent` with a separate order collection period.
- At auction end, compute clearing price via maximum-volume matching.
- Transition to continuous trading after open auction; run close auction before market close.
- Significant effort — affects Exchange state machine, message flow, and all agent timing.

### 16. Register Gym Agents in Config System

**Type:** Housekeeping | **Effort:** Small

**Problem:** `CoreBackgroundAgent` and `FinancialGymAgent` from `abides-gym` are not registered in the declarative config system.

**Motivation:** Completeness. Users building gym environments via the config system cannot reference these agents declaratively.

**Implementation hints:**
- Add `@register_agent` decorators in `abides_gym` or add a `gym_registrations.py` in the config system that conditionally imports from `abides_gym`.

### 17. Remove Deprecated MeanRevertingOracleConfig

**Type:** Housekeeping | **Effort:** Small

**Problem:** The deprecated `MeanRevertingOracle` (non-sparse) is still compilable via `MeanRevertingOracleConfig`. It carries a `DeprecationWarning` and a step-count safety guard, but remains a foot-gun.

**Motivation:** Users may accidentally select the non-sparse oracle, which pre-computes the entire price path in memory. The sparse oracle is strictly superior. Removing the config eliminates confusion.

**Implementation hints:**
- Remove `MeanRevertingOracleConfig` from `models.py` and the compiler's oracle dispatch.
- Keep the `MeanRevertingOracle` class itself (for direct low-level use) but remove its config-system entry point.
- Update docs and templates if any reference it.

---

## P2 — Simulation Science

This section covers the infrastructure for understanding, validating, and calibrating simulations. Items are ordered by dependency: #18 (ensemble runner) is the foundation; #19–#22 build on it.

```
#18 ensemble runner
    ├── #19 stylized facts scorecard
    ├── #20 sensitivity analysis
    └── #22 agent impact attribution

#18 + #19 + #20 → #21 scenario calibrator
```

When implemented, these utilities live in `abides-markets/abides_markets/simulation/science.py` (or a `science/` subpackage).

### 18. Multi-Seed Ensemble Runner

**Type:** Infrastructure | **Effort:** Small

**Problem:** Running a single simulation produces stochastic output — a single result is not a reliable estimate of a configuration's behaviour. There is no built-in utility to run the same config over multiple seeds and aggregate results.

**Motivation:** Foundation for all other simulation-science items. Also directly useful for researchers who want confidence intervals on market quality metrics.

**Implementation hints:**
- `run_ensemble(config, n_seeds, n_jobs=-1, profile=ResultProfile.SUMMARY) -> EnsembleResult`.
- Use `multiprocessing.Pool` (consistent with `HASUFEL_PARALLEL_SIMULATION.md`). Each worker calls `run_simulation(config.with_seed(seed))` with an independent seed.
- `EnsembleResult` wraps a `pd.DataFrame` (one row per seed, one column per scalar metric) and exposes `.mean()`, `.std()`, `.ci(alpha=0.05)` helpers.
- RNG safety: derive per-seed configs using `SimulationConfig.with_seed(seed)` so master-seed isolation is preserved.

### 19. Stylized Facts Scorecard

**Type:** Validation | **Effort:** Medium | **Depends on:** #18

**Problem:** There is no programmatic way to measure whether a simulation produces realistic market microstructure. Currently, realism is assessed by visual inspection of metric printouts.

**Motivation:** The empirical microstructure literature (Cont 2001) defines ~8 canonical stylized facts present in all liquid markets. A scorecard that checks them gives a single, reproducible quality signal — both for human review and as the objective function for the calibrator (#21).

**Implementation hints:**
- `score_stylized_facts(ensemble: EnsembleResult) -> StyleFacts` computing:
  - **Return kurtosis** > 3 (fat tails)
  - **Near-zero return autocorrelation** at lags 1–10 (efficient market)
  - **Positive autocorrelation of absolute returns** at lags 1–20 (volatility clustering)
  - **Negative volume–spread correlation** (spread widens in thin markets)
  - **Bid-ask spread distribution** shape (right-skewed, power-law tail)
- Each fact produces a pass/fail flag and a numeric score. `StyleFacts.composite_score` is the weighted mean (weights configurable).
- Requires `ResultProfile.QUANT` for return series. Use the L1 mid-price series for return computation.
- `StyleFacts` should be serialisable (`model_dump()`) for logging and comparison.

### 20. Parameter Sensitivity Analysis

**Type:** Analysis | **Effort:** Medium | **Depends on:** #18

**Problem:** It is unknown which parameters most strongly influence simulation output metrics. Tuning is done by intuition, which scales poorly.

**Motivation:** A sensitivity analysis identifies which parameters to focus on during calibration and which are near-irrelevant, reducing the calibrator's search dimensionality.

**Implementation hints:**
- `sensitivity_analysis(base_config, param_space, metric_keys, n_samples, n_seeds) -> SensitivityResult`.
- `param_space`: dict mapping dotted parameter paths (e.g. `"noise_agent.wake_up_freq"`) to `(low, high)` ranges.
- Sample the space via Latin Hypercube Sampling (`scipy.stats.qmc.LatinHypercube`). For each sample, patch the config, run the ensemble, record metric means.
- Compute first-order sensitivity indices as Pearson correlation between parameter values and metric means across samples.
- `SensitivityResult.matrix` is a `DataFrame[param × metric]` of correlation coefficients; expose `.plot_heatmap()` convenience method.

### 21. Scenario Calibrator

**Type:** Optimisation | **Effort:** Large | **Depends on:** #18, #19, #20

**Problem:** Finding a configuration that produces target market behaviour (e.g., a specific spread, volatility, or volume regime) requires manual trial-and-error.

**Motivation:** A calibrator makes the simulator usable as a model of a specific market. Two modes are supported:
- **Stylized-facts mode**: find config that maximises `StyleFacts.composite_score` subject to optional metric constraints.
- **Historical mode**: find config that minimises distance between simulated metric vector and an empirical metric vector derived from real L1/L2 data via the existing `BatchDataProvider` protocol.

**Implementation hints:**
- `calibrate(base_config, param_space, target, n_seeds, optimizer) -> CalibrationResult`.
- `target` is either a `StyleFacts` target score or a `dict[str, float]` of metric targets extracted from historical data.
- Objective function: `loss(params) = ensemble_mean_metric_distance(patched_config, target)`. Evaluated via `run_ensemble()`.
- Optimizers: `"nelder_mead"` (fast, derivative-free) or `"differential_evolution"` (global, slower) via `scipy.optimize`.
- `param_space` doubles as bounds for the optimizer.
- Use sensitivity analysis (#20) output to reduce `param_space` to the top-K most influential parameters before running the optimizer — document this as the recommended workflow.
- `CalibrationResult` includes: best config, best loss, convergence trace, and a `StyleFacts` score of the calibrated result.

### 22. Agent Impact Attribution

**Type:** Analysis | **Effort:** Small | **Depends on:** #18

**Problem:** It is not clear which agent types drive which market-quality outcomes. Debugging misconfigured simulations requires guesswork.

**Motivation:** Ablation studies identify causal relationships between agent populations and market phenomena (e.g., "removing noise agents collapses spread", "adding AMM halves no-bid time"). Useful for both model validation and debugging.

**Implementation hints:**
- `agent_impact(base_config, metric_keys, n_seeds) -> ImpactResult`.
- For each agent group in the config: run the ensemble with that group removed (count=0) and with count halved.
- `ImpactResult.delta_matrix` is a `DataFrame[agent_group × metric]` of absolute and relative metric changes vs. baseline.
- Expose `.rank_by_impact(metric)` to surface the most influential agent group for a given metric.
- Implementation note: "removing" a group means patching count to 0 via the same config-patching mechanism used by #20.

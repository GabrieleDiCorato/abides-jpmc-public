2026-03 Release v2.5.0
==================

New Agents
----------

* **MeanReversionAgent** — contrarian strategy that trades on z-score
  deviations from a rolling mean.  Buys when price is significantly below
  the mean, sells when above.  Configurable lookback, entry/exit thresholds,
  and position sizing.  Registered as ``mean_reversion`` (category: strategy).

* **TWAPExecutionAgent** — time-weighted average price execution.  Slices a
  parent order into equal-sized child IOC orders at regular intervals across
  the execution window.  Shares ``BaseSlicingExecutionAgent`` base class with
  POV.  Registered as ``twap_execution`` (category: execution).

* **VWAPExecutionAgent** — volume-weighted average price execution.  Accepts
  a configurable intraday volume profile and sizes each child slice
  proportional to the expected volume in each bucket.  Registered as
  ``vwap_execution`` (category: execution).

Execution Agent Infrastructure
------------------------------

* **BaseSlicingExecutionAgent** — extracted from POVExecutionAgent as a shared
  base for all slicing execution agents (POV, TWAP, VWAP).  Handles common
  concerns: execution window management, IOC child order submission, fill
  tracking, arrival-price capture, and summary logging.

* **NoiseAgent multi-wake mode** — ``NoiseAgent`` can now be configured for
  continuous wakeups (``multi_wake=True``), producing a steady background
  noise flow throughout the session instead of a single trade-and-sleep.

Order Types
-----------

* **Time-in-Force qualifiers** — ``LimitOrder`` now supports ``IOC``
  (immediate-or-cancel), ``FOK`` (fill-or-kill), and ``DAY`` (cancel at
  close) via the ``TimeInForce`` enum.  The exchange enforces each qualifier
  at order entry and at end-of-day.

* **Stop orders** — new ``StopOrder`` type with exchange-side trigger logic.
  When the market price crosses the stop price, the exchange converts the
  stop into a limit or market order.  ``TradingAgent.place_stop_order()``
  provides the high-level API.

Market Maker Enhancements
-------------------------

* **End-of-day position flatten** — ``AdaptiveMarketMakerAgent`` now
  aggressively flattens its inventory near market close, reducing overnight
  risk.  Logged as ``AMM_FLATTEN`` events.

Data Extraction & Analytics
---------------------------

* **Execution-quality metrics** — ``ExecutionMetrics`` model captures
  VWAP slippage, participation rate, implementation shortfall, and
  arrival-price comparison for every execution-category agent.

* **Causal trade attribution** — ``TradeAttribution`` model links each
  execution to its passive and aggressive orders, enabling post-simulation
  market-impact analysis.  Extracted via ``ResultProfile.TRADE_ATTRIBUTION``.

* **Equity curves** — ``EquityCurve`` model built from ``FILL_PNL`` log
  events provides per-fill NAV time-series and max-drawdown computation.
  Extracted via ``ResultProfile.EQUITY_CURVE``.

Bug Fixes
---------

* **Fixed: execution agents killed simulations** —
  ``BaseSlicingExecutionAgent.wakeup()`` returned ``bool`` instead of
  ``None``, triggering the kernel's gym-agent interrupt mechanism and
  terminating the simulation after ~2 wakeups.  All execution agents
  (TWAP, VWAP, POV) were completely non-functional.

* **Fixed: OrderBook EXEC history stored price=None** — non-PTC trade
  executions recorded ``None`` for price in history entries, breaking
  trade attribution and equity curve extraction.  Changed to always
  use ``fill_price``.

* **Fixed: execution agent default offsets too large** — default
  ``start_time_offset`` and ``end_time_offset`` of 30 minutes consumed
  the entire rmsc04 market window.  Changed to 5 minutes.

Tests & Tooling
---------------

* **972 tests** — up from ~762 in v2.4.0.  New test files cover order book
  invariants, PTC edge cases, thin-market scenarios, execution-quality
  numerics, market boundary behavior, oracle numerics, risk interactions,
  and TradingAgent async gotchas.

* **Full-feature evaluation script** — ``version_testing/evaluate_all_agents.py``
  runs 10 scenarios covering all templates, agent types, and data extraction
  profiles.


2026-03 Release v2.4.0
==================

Seed Derivation — Composition-Invariant RNG Hierarchy
-----------------------------------------------------

**Breaking change:** the compiler now derives per-component seeds via
SHA-256 identity hashing instead of a sequential ``master_rng.randint()``
chain.  This means that **for the same master seed integer, agents will
receive different ``random_state`` objects than in v2.3.0.**  Simulation
results generated with earlier versions are not reproducible under v2.4.0.

This is a deliberate departure from the sequential-draw approach described
in the original ABIDES literature.  The new scheme provides two guarantees
that the old one could not:

1. **Order independence** — declaring agent groups in a different order
   (e.g. ``enable_agent("value", ...)`` before ``enable_agent("noise", ...)``)
   produces identical seeds.  A Pydantic ``model_validator`` on
   ``SimulationConfig`` now sorts agent groups by name at construction time.

2. **Composition invariance** — adding, removing, or resizing an agent
   group does not shift the seeds of any other component.  Each component's
   seed depends only on ``(master_seed, component_name, index)`` via
   ``hashlib.sha256(f"{seed}:{component}:{index}")``, never on what other
   components exist.

Practical impact: researchers can inject a new strategy into an existing
configuration and be certain that all *other* agents behave identically
to the baseline — a prerequisite for controlled A/B experiments.

* **``_derive_seed()`` helper** — new function in ``compiler.py`` that maps
  ``(master_seed, component, index)`` → unsigned 32-bit integer via SHA-256.
* **Sequential ``master_rng`` removed** — the compiler no longer maintains
  a shared ``np.random.RandomState`` for seed allocation.  Oracle, exchange,
  each agent group, kernel, and latency model each derive their seed
  independently.
* **``_sort_agents`` model validator** — ``SimulationConfig`` sorts
  ``self.agents`` alphabetically after construction, ensuring canonical
  ordering for deterministic agent-ID assignment and YAML serialization.
* **5 new / strengthened tests** — order-independence, oracle injection,
  adding-agent-preserves-seeds, changing-count-preserves-seeds, and
  strengthened determinism assertion (verifies full ``random_state``
  internal state equality).
* **Documentation** — ``PARALLEL_SIMULATION_GUIDE.md`` updated with the
  new RNG hierarchy diagram and composition-invariance note.


2026-03 Release v2.3.0
==================

Config UX — Human-Readable Units
---------------------------------

Config fields that previously required obscure nanosecond values or
per-nanosecond scientific-notation rates now accept **duration strings**
(e.g. ``"1min"``, ``"175s"``, ``"48d"``).  Internal constructors and
the oracle still use per-nanosecond values — only the config layer changed.

* **``order_rate_window``** (was ``order_rate_window_ns: int``).
  ``BaseAgentConfig`` field now accepts a duration string (default ``"1min"``).
  Converted to nanoseconds when building ``RiskConfig``.

* **``mean_wakeup_gap``** (was ``lambda_a: float``).
  ``ValueAgentConfig`` field now accepts a duration string (default ``"175s"``).
  Converted to a Poisson arrival rate (``1/ns``) in ``_prepare_constructor_kwargs``.

* **``mean_reversion_half_life``** (was ``kappa: float``).
  Both ``SparseMeanRevertingOracleConfig`` and ``ValueAgentConfig`` now
  accept a duration string (oracle default ``"48d"``, agent default ``None``
  = auto-inherit).  Converted to per-nanosecond kappa via ``ln(2)/ns``.

* **``subscribe_freq``** — ``AdaptiveMarketMakerConfig`` field changed from
  ``int`` (nanoseconds) to duration string (default ``"10s"``).

* **``megashock_mean_interval``** (was ``megashock_lambda_a: float``).
  ``SparseMeanRevertingOracleConfig`` field now accepts a duration string
  or ``None`` to disable megashocks (default ``"100000h"`` ≈ 11.4 years).
  Converted to a Poisson rate in the compiler.

* **``fund_vol`` description improved** — now explains per-√(ns) units and
  practical impact on daily price variation.

* Templates and documentation updated to use new field names.


2026-03 Release v2.2.1
==================

Code Quality
------------

* **snake_case method renames** — replaced remaining CamelCase methods across
  all agents: ``placeOrder`` → ``place_order`` (NoiseAgent, ValueAgent),
  ``updateEstimates`` → ``update_estimates`` (ValueAgent),
  ``logL2style`` → ``log_l2_style`` (ExchangeAgent).
* **VALID_STATES runtime validation** — ``TradingAgent`` gained a ``state``
  property that validates assignments against a ``VALID_STATES`` frozenset.
  Subclasses that declare ``VALID_STATES`` get instant ``ValueError`` on
  typos; agents using non-string state (AdaptiveMarketMakerAgent) are
  unaffected (``VALID_STATES = None`` skips validation).
  Declared on NoiseAgent, ValueAgent, MomentumAgent, POVExecutionAgent.
* **Standardized ``@register_agent``** — ``builtin_registrations.py``
  converted from ``registry.register()`` to the ``register_agent()``
  decorator pattern, matching the public API documented in the guide.

Documentation
-------------

* **NoiseAgent annotated as reference implementation** — full rewrite with
  WHY comments explaining event-driven architecture, async data flow,
  state machine pattern, ``super().wakeup()`` guard, and ``isinstance``
  dispatch.
* **Auto-wired constructor params documented** — ``TradingAgent`` docstring
  now lists the 6 parameters auto-injected by the config system.
* **Custom Agent Guide expanded** — added §9 (copy-paste agent scaffold
  with TODO markers), §10 (10-step agent-building checklist), and §11
  (testing section with ``make_agent()`` and ``SimulationBuilder`` patterns).

Tests
-----

* Created shared ``conftest.py`` with ``make_agent()`` helper,
  ``StubKernel`` stand-in, and ``rmsc04_config`` session fixture.
* Updated all test references for snake_case method renames.


2026-03 Release v2.2.0
==================

Risk Management
---------------

* **Position limits** — agents can now enforce per-symbol position caps.
  A symmetric ``[-N, +N]`` share limit is checked before every order placement
  (limit, market, multi-leg, and replace). Two enforcement modes:
  reduce-only (clamp the order to the remaining headroom) or hard-block
  (reject the order outright).
* **Circuit breaker** — agents automatically stop trading when a drawdown
  threshold or order-rate limit is breached. The kill-switch is latching
  (once tripped it stays tripped) and is also checked proactively on every
  fill, so a sudden adverse move disables the agent immediately.
* **RiskConfig object** — the five risk parameters (position limit, clamp mode,
  max drawdown, max order rate, rate window) are bundled in a frozen
  ``RiskConfig`` dataclass.  It is wired through every concrete agent class
  and auto-assembled by the config system, so risk rules declared in YAML
  reach the agent without manual plumbing.
* **Per-fill P&L tracking** — ``TradingAgent`` now logs a ``FILL_PNL`` event
  after every execution with a running NAV and peak-NAV high-water mark,
  enabling post-simulation equity-curve analysis.

Oracle Redesign
---------------

* **Oracle is now a required configuration choice** — ``MarketConfig.oracle``
  has no default. Every simulation must explicitly set an oracle or pass
  ``oracle: null`` (oracle-absent mode). This eliminates silent fallback
  behaviour.
* **Oracle-absent mode** — setting ``oracle: null`` plus an explicit
  ``opening_price`` (integer cents) produces a valid simulation without a
  fundamental value process. Useful for replay and execution-only scenarios.
* **ValueAgent auto-inherits oracle parameters** — ``r_bar``, ``kappa``, and
  ``sigma_s`` are pulled from the oracle config automatically when not
  overridden, removing a common source of parameter mismatch.
* **ExternalDataOracle injection** — ``ExternalDataOracleConfig`` is now a
  pure marker type (no ``data_path``). Users build the oracle instance
  externally and inject it via ``builder.oracle_instance()`` or
  ``compile(config, oracle_instance=...)``.
* **Compile-time validation** — the compiler rejects ValueAgent without an
  oracle, and the builder raises ``ValueError`` when extra kwargs are passed
  alongside ``oracle(type=None)``.

Order Management
----------------

* **Replace-order support** — ``ValueAgent`` and ``AdaptiveMarketMakerAgent``
  now use ``replace_order()`` to amend existing orders in-place instead of
  cancel-then-resubmit, cutting message traffic on every re-quote cycle.

Config System Enhancements
--------------------------

* **Cross-agent validation** — ``SimulationBuilder.validate()`` returns a
  structured ``ValidationResult`` that checks inter-agent consistency
  (e.g. a ValueAgent without an oracle, inverted time windows, missing
  exchange).
* **Agent registry metadata** — each registered agent type now carries
  oracle/count/dependency metadata and a category taxonomy (background,
  execution, market-maker, …). A full manifest API is available for
  programmatic discovery.
* **Richer field descriptions** — config model fields carry human-readable
  descriptions, physical-unit annotations, and Pydantic validators, making
  schema introspection and AI-assisted configuration more reliable.
* **Model-level validators** — ``MarketConfig`` rejects invalid combinations
  (oracle absent without ``opening_price``, ``start_time ≥ end_time``) at
  construction time rather than at compile time.
* **Time-window inversion guards** — ``NoiseAgentConfig`` and
  ``POVExecutionAgentConfig`` factories reject inverted wakeup/execution
  windows during ``_prepare_constructor_kwargs()``.
* **Exposed hidden agent parameters** — ``AdaptiveMarketMakerConfig`` gained
  ``anchor``, ``subscribe``, ``subscribe_freq``, ``subscribe_num_levels``,
  ``min_imbalance``; ``MomentumAgentConfig`` gained ``subscribe``.
* **Compiler error context** — when agent instantiation fails, the error
  message now includes the ``agent_type_name`` that caused the failure.
* **Composable ``_EXCLUDE_FROM_KWARGS``** — risk-field exclusion sets use a
  shared ``_BASE_EXCLUDE`` constant, so adding a new risk field propagates
  to all subclass configs automatically.

Constructor ↔ Config Alignment
------------------------------

* Aligned ``AdaptiveMarketMakerAgent`` constructor defaults to rmsc04-tuned
  values (7 parameters updated).
* Aligned ``ValueAgent.lambda_a`` default to the rmsc04 Poisson rate
  (``5.7e-12``), replacing a stale upstream value.

Simulation Analytics
--------------------

* **VWAP and execution metrics** — ``SimulationResult`` now exposes a
  ``summary_dict()`` method and ``ExecutionMetrics`` with VWAP computation,
  ready for dashboard consumption.

Performance
-----------

* **O(1) message dispatch** — ``TradingAgent`` and ``ExchangeAgent`` replaced
  their ``isinstance`` dispatch chains with dictionary-based dispatch, making
  ``receive_message()`` constant-time in the number of message types.
* **MRO-aware cached dispatch** — the dispatch table respects class
  hierarchies, so subclass handler overrides are resolved correctly.
* **Bounded collections** — ``MomentumAgent``'s mid-price history switched to
  ``deque(maxlen=N)``; ``SparseMeanRevertingOracle``'s ``f_log`` is bounded
  at 100 000 entries. Both prevent unbounded memory growth in long
  simulations.
* **Configurable MA windows** — ``MomentumAgent`` gained ``short_window`` and
  ``long_window`` parameters so the moving-average look-back is no longer
  hard-coded.
* **Subclass-safe type checks** — ``NoiseAgent`` and ``ValueAgent`` replaced
  ``type(self) is ...`` with ``isinstance`` for proper subclass
  compatibility.
* **Removed redundant deepcopy** on limit-order handling in
  ``ExchangeAgent``.

Bug Fixes
---------

* Fixed ``TradingAgent`` ``kernel_stopping`` result-accumulation pattern:
  replaced ad-hoc ``if/else`` dict building with ``defaultdict(int)`` in
  Kernel.
* Fixed ``NoiseAgent`` surplus calculation to use integer-cent arithmetic
  throughout (floor division instead of float division).
* Fixed ``MomentumAgent`` crossover trigger comparison (``>`` → ``>=``).
* Fixed ``AdaptiveMarketMakerAgent`` ``subscribe_freq`` type annotation
  (``float`` → ``int`` nanoseconds).

Tests
-----

* 28 circuit-breaker tests (drawdown, rate, latching, tumbling window, fill
  trip, config propagation)
* 44 position-limit tests (pending delta, enforcement modes, all order paths,
  config fields)
* 27 replace-order regression tests (ask-side, crossing, non-existent, AMM
  diff-and-replace, ValueAgent partial-fill)
* 16 oracle-redesign tests (oracle-absent, auto-inheritance, builder API)
* 22 RiskConfig tests (dataclass, unpacking precedence, fill P&L, agent
  forwarding, config injection)
* Config-system integrity tests (constructor alignment, kwarg rejection,
  model validators, time guards, compiler context)
* Input-validation tests for ``MomentumAgent`` and ``ValueAgent`` parameter
  guards

Documentation
-------------

* Complete Custom Agent Implementation Guide rewrite with adapter pattern,
  ``RiskConfig``, ``replace_order``, and config-system integration
* Updated Copilot instructions with oracle system rules and custom-agent
  pattern
* Expanded Config System guide with Oracle Configuration section and examples


2026-03 Release v2.1.0
==================

Code Quality
------------

* Modernised all type annotations across codebase: ``List`` → ``list``,
  ``Dict`` → ``dict``, ``Optional[X]`` → ``X | None``, ``Tuple`` → ``tuple``
* Replaced all ``.format()`` string formatting with f-strings
* Resolved all CI checks: ruff linting, isort import ordering, black formatting,
  and pyright/mypy type checking
* Added ``# type: ignore`` annotations for mypy false positives on
  ``importlib.util`` usage

Documentation
-------------

* Added technical README files for abides-core, abides-markets, and abides-gym modules
* Updated AGENT_ASSESSMENT.md for v2.0.0: removed resolved issues, verified
  remaining open items, added product evaluation section
* Added professional CI/test/license badges to project README

Housekeeping
------------

* Removed tracked ``.DS_Store`` files and added pattern to ``.gitignore``
* Cleaned up folder layout sections from module-level READMEs


2026-03 Release v2.0.0
==================

Agent Bug Fixes
---------------

* Fixed AdaptiveMarketMakerAgent subscribe-mode crash: `self.state["MARKET_DATA"]`
  referenced a key that never existed — corrected to `self.state["AWAITING_MARKET_DATA"]`
* Fixed ValueAgent `log_orders` type annotation: was `float`, corrected to `bool`
* Fixed MomentumAgent `KeyError` on empty order book: replaced bare dict access
  `self.known_bids[self.symbol]` with safe `.get(self.symbol, [])`

Quant Review Fixes
------------------

* Fixed `TradingAgent.get_known_bid_ask()` return type: `float | None` corrected to
  `int | None` — prices are always integer cents
* Fixed `TradingAgent.get_known_bid_ask_midpoint()` `KeyError` on missing symbol:
  bare `self.known_bids[symbol]` replaced with safe `.get(symbol, [])`
* Fixed `TradingAgent.mark_to_market()` `KeyError` on missing last trade: two bare
  `self.last_trade[symbol]` accesses replaced with safe `.get(symbol)` with None guard
* Removed dead `if log_orders is None:` block in `TradingAgent.__init__()` — the
  constructor parameter already defaults to `True`, this branch was unreachable
* Fixed `ExchangeAgent` duplicate `isinstance(data_sub, self.L3DataSubscription)`
  check in `publish_data_to_subscribers()`
* Fixed `ExchangeAgent.metric_trackers` crash when `use_metric_tracker=False`:
  attribute now always initialised (as empty dict when unused)
* Normalised `ValueAgent` buy/sell direction variable from `int` (0/1) to `bool`;
  replaced `buy == 1` comparison with truthiness check
* Normalised `NoiseAgent` buy/sell direction: renamed `buy_indicator` to `buy`,
  converted from `int` to `bool`
* Fixed `MomentumAgent` MA values stored as floats via `.round(2)`: replaced with
  `int(round(...))` to maintain integer-cent price convention
* Fixed `AdaptiveMarketMakerAgent` poll-mode wakeup discarding `initialise_state()`
  return value — result now assigned to `self.state`
* Fixed `POVExecutionAgent` `last_bid`/`last_ask` type annotations from `float` to
  `int` — prices are always integer cents

Oracle Fixes
------------

* Added `f_log` class attribute to Oracle ABC — subclasses that don't track
  fundamental history now return `{}` by default instead of raising `AttributeError`
* Replaced fragile `hasattr(self.oracle, "f_log")` check in ExchangeAgent with
  truthiness check `if self.oracle.f_log:` — works correctly with the new default
* Removed dead `self.oracle = self.kernel.oracle` line in NoiseAgent (unreachable code)

MeanRevertingOracle Safety
--------------------------

* Added `DeprecationWarning` — users are directed to `SparseMeanRevertingOracle`
* Added `ValueError` guard rejecting time ranges > 1 000 000 steps to prevent
  accidental multi-GB memory allocation

Tests
-----

* Added `test_agent_fixes.py` — 16 regression tests covering all agent and oracle fixes
* Added 3 safety tests to `test_mean_reverting_oracle.py` for deprecation warning
  and step-count guard

2026-03 Release v1.3.0
==================


Declarative Configuration System
---------------------------------

* Added pluggable, AI-friendly config system built on Pydantic models
* `SimulationBuilder` fluent API with composable templates (rmsc04, liquid_market, thin_market)
* Agent registry with `@register_agent` decorator for third-party agent types
* YAML/JSON serialization via `save_config()` / `load_config()`
* Per-agent-type computation delays (overrides the global default per agent group)
* AI discoverability API: `list_agent_types()`, `get_config_schema()`, `validate_config()`
* Compiler produces the same runtime dict format as `build_config()` — fully backward compatible

Simulation Runner
-----------------

* Promoted `run_simulation(config)` as the primary API for running simulations —
  compiles a fresh runtime dict internally and returns a typed, immutable `SimulationResult`
* `SimulationConfig` is immutable and reusable: the same config can be passed to
  `run_simulation()` any number of times with identical results
* `run_batch(configs)` for parallel multi-simulation execution via `multiprocessing`
* Removed deep-copy from `abides.run()` — the low-level `compile()` → `abides.run()`
  path now consumes the runtime dict once (original ABIDES behaviour restored)

Bug Fixes
---------

* Fixed `Agent.get_computation_delay()` calling nonexistent `Kernel.get_agent_compute_delay()` — added the missing Kernel method
* Added `per_agent_computation_delays` support to Kernel for declarative per-agent delay configuration
* Fixed `@register_agent` decorator raising `ValueError` when a notebook cell that defines
  a custom agent is re-executed — the decorator now silently overwrites the previous
  registration (`allow_overwrite=True` by default).
* Fixed `_register_builtins()` guard: previously skipped all built-in registration if *any*
  agent was already in the registry (e.g. a custom agent registered before import-time
  builtins ran).  Guard now checks specifically for the five built-in names.


Project Rename
--------------

* Renamed project from `abides-jpmc-public` to `abides-hasufel`
* Updated all documentation, metadata, and references to reflect the new project identity
* Added derivative-work copyright for Gabriele Di Corato to LICENSE
* Upstream attribution to Georgia Tech (original ABIDES) and J.P. Morgan Chase preserved


2026-03 Release v1.2.0
==================


Bugs Fixed
----------

* Fixed latency matrix row aliasing — `[[v]*N]*N` creates N references to the same inner list; replaced with `[[v]*N for _ in range(N)]`
* Fixed `MessageBatch` computation delay applied N times per batch instead of once
* Fixed `get_l1_bid_data()` returning the wrong price level after skipping zero-quantity levels; added bounds check


Performance Improvements
------------------------

* Order book insert, cancel, modify, and partial-cancel operations replaced O(N) linear scan with O(log N) binary search via `bisect`
* Subscription publishing scoped to the affected symbol only — previously iterated all symbols on every order event


Other Changes
-------------

* Removed 6 redundant `deepcopy()` calls from exchange agent order processing
* Replaced `filter+lambda` with list comprehension in L2 book data methods
* Changed `logEvent` default to `deepcopy_event=False`; added explicit `deepcopy_event=True` where holdings dict is logged
* Replaced `queue.PriorityQueue` with `heapq` in Kernel message queue, eliminating mutex overhead in the single-threaded event loop
* Fixed subscription cancel list mutation during iteration
* Removed global `pd.set_option("display.max_rows", 500)` from exchange agent module scope
* Documentation: cleaned up and simplified all docs to match current code state (removed outdated concurrency appendix, condensed remediation plan to changelog table)


2026-01 Release v1.1.0
==================


Other Changes
-------------

* Added the POVExecutionAgent implementation, which was referenced but missing, to restore RMSC03 env
* Fixed the version_testing suite of non-regression tests
    * Removed hardcoded comparison between specific commits
    * Restored logging to enable extracting runtime metrics for testing


Dependency Updates
-------------

This fork modernizes the archived `abides-jpmc-public` project with updated dependencies for compatibility with current Python ecosystems.

### Core Dependencies

| Package | Original Version | Updated Version | Notes |
|---------|-----------------|-----------------|-------|
| coloredlogs | 15.0.1 | ≥15.0.1 | Maintained compatibility |
| gym | 0.18.0 | **gymnasium ≥1.0.0** | **Migrated to maintained fork** |
| numpy | 1.22.0 | **≥2.0.0** | **Major version update** |
| pandas | 1.2.4 | **≥2.2.0** | **Major version update** |
| pomegranate | 0.14.5 | **Removed** | **Deprecated dependency** |
| psutil | 5.8.0 | ≥6.0.0 | Major version update |
| ray[rllib] | 1.7.0 | **≥2.40.0** | **Major version update** |
| scipy | 1.10.0 | **≥1.14.0** | Minor version update |
| tqdm | 4.61.1 | ≥4.67.0 | Minor version update |
| p_tqdm | 1.3.3 | ≥1.4.0 (dev) | Moved to optional dev dependencies |
| matplotlib | N/A | **≥3.9.0** | **New dependency added** |

### Development Dependencies

| Package | Original Version | Updated Version | Notes |
|---------|-----------------|-----------------|-------|
| pre-commit | 2.13 | ≥4.0.0 | Major version update |
| pytest | 6.2.4 | ≥8.3.0 | Major version update |
| pytest-cov | 2.12.1 | ≥6.0.0 | Major version update |
| sphinx | 3.5.4 | ≥8.0.0 | Major version update |
| sphinx-autodoc-typehints | 1.12.0 | ≥2.5.0 | Major version update |
| sphinx-book-theme | 0.0.42 | ≥1.1.0 | Major version update |

### Breaking Changes

- **gym → gymnasium**: OpenAI Gym is deprecated. All code using `gym` has been migrated to `gymnasium` (the maintained fork).
- **pomegranate removed**: This probabilistic modeling library is no longer actively maintained and has been removed.
- **numpy 2.0**: Includes breaking changes in the C API and some deprecated functions removed.
- **pandas 2.x**: Various API changes and performance improvements.




2021-10-15 Release
==================

New Features
------------

- WandB + rllib custom metrics + background V2 + autocalib (PR #86)

Other Changes
-------------

- Code cleanup for open source release (PR #87, #88, #90)
- Separation Gym core into markets and true gym core (PR #91)


2021-09-28 Release
==================

New Features
------------

- Order book state subscriptions and alerts (PR #73)
- ABIDES-gym (PRs #77, #79, #82)
- Event data subscriptions (PR #81)
- Background Agents v2 + wandB scripts (PR #85)

Other Changes
-------------

- Optimise agent event log data structures for memory use (PR #74)
- Feature marketreplay realdata2 (PR #75)
- Replace use of pd.Timestamp with raw ints (PR #76)
- Simplify kernel initialisation (PR #80)
- Add message batches (PR #83)

Bugs Fixed
----------

- Fix debug message for modify order msg (PR #72)
- Fix Order Book Imbalance subscription (PR #78)


2021-07-27 Release
==================

New Features
------------

- Add price to comply order types (PR #64)
- Add insert by ID option for limit orders (PR #68)

Other Changes
-------------

- Improve abides-cmd and config layout (PR #62)

Bugs Fixed
----------

- Fix the place order of MM to not place order of size 0 when backstop qty = 0 (PR #63)
- Fix attempts to cancel MarketOrders (PR #65)
- Remove placing orders when size is 0 (PR #66)
- Add test for negative price on limit orders (PR #67)
- Fix regression tests (PR #69)
- Fix r_bar types (PR #70)
- Fix end of day mark to market (PR #71)


2021-06-29 Release
==================

New Features
------------

- Add hidden orders to order book (PR #56)
- Use built-in Python logging library (PR #57)

Other Changes
-------------

- Reorganise directory layout (PR #54)
- Add initial pre-commit hooks and RMSC unit test (PR #58)
- Update generated documentation to work with refactored code layout (PR #59)
- Replace is_buy_order flag with Side.BID or Side.ASK enum (PR #60)

Bugs Fixed
----------

- Correcting markToMarket function to multiply by shares (PR #61)


2021-06-15 Release
==================

New Features
------------

- Add transacted volume and L1 and L3 data subscriptions (PR #42)
- Change message types to use dataclasses (PR #45)
- Add replaceOrder command to OrderBook (PR #48)

Other Changes
-------------

- Refactor subscription message and data classes (PR #42)
- Simplify market order handling code by removing limit order creation (PR #43)
- Use a flat data structure to store order history (PR #44)
- Simplify handleLimitOrder (PR #46)
- Refining message stream logging and orderbook log2 (PR #47)
- Add more unit tests for OrderBook (PR #49)

Bugs Fixed
----------

- Fix NoiseAgent and Value agent random seed sources (PR #50)
- Fix various Message class issues (PR #51)


2021-06-01 Release
==================

New Features
------------

- Initial OrderBook unit tests (PR #39)
- New order book data getting methods (L1, L3 data) (PR #40)

Other Changes
-------------

- OrderBook history and tracking now optional (PR #27)
- Removed unused ExchangeAgent code (PR #30)
- Reduce number of Python deepcopies (PR #32)
- General tidy of Order classes (PR #38)
- Add warning if invalid arguments passed to abides_cmd (PR #41)

Bugs Fixed
----------

- Fix RMSC03 spec documentation (PR #35)
- Fix version testing script timer (PR #36)


2021-05-18 Release
==================

New Features
------------

- Automatically generated documentation using Sphinx-Doc (PR #18)
- Developer guide in documentation with best practices (PR #18)

Other Changes
-------------

- Improved documentation strings in code (PR #18)
- Type annotations added for most files (PR #18)
- Faster deepcopy of orders (PR #21)
- Simplified order ID generation (PR #25)
- Agent class 'type', 'name' and 'RandomState' parameters now optional (PR #26)

Bugs Fixed
----------

- Correction of type errors in Noise agent (PR #23)
- Noise calculation error fix (PR #17)

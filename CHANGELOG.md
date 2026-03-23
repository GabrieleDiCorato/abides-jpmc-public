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

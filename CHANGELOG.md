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

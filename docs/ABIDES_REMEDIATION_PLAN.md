# ABIDES Remediation Changelog

All items completed. 165 tests passing (159 original + 6 new regression tests).

| # | Category | Change | File(s) | Commit |
|---|----------|--------|---------|--------|
| 1.1 | **Correctness** | Fixed latency matrix row aliasing (`[[v]*N]*N` → comprehension) | `kernel.py` | `5612701` |
| 1.2 | **Correctness** | Fixed MessageBatch computation delay applied N times | `kernel.py` | `5612701` |
| 1.3 | **Correctness** | Fixed `get_l1_bid_data()` returning wrong price level | `order_book.py` | `eeef5a6` |
| 2.1 | **Performance** | Order book insert/cancel/modify → O(log N) with `bisect` | `order_book.py` | `6f5fed4` |
| 2.2 | **Performance** | Subscription publishing scoped to affected symbol | `exchange_agent.py` | `a74c4c8` |
| 3.1 | **Memory** | Removed 6 redundant `deepcopy()` calls in exchange agent | `exchange_agent.py` | `604b6f4` |
| 3.2 | **Memory** | `filter+lambda` → list comprehension in L2 data methods | `order_book.py` | `7cf2320` |
| 3.3 | **Memory** | `logEvent` default changed to `deepcopy_event=False` | `agent.py`, `trading_agent.py` | `1377e72` |
| 4.1 | **Quality** | Fixed list mutation during iteration in subscription cancel | `exchange_agent.py` | `f1a296a` |
| 4.2 | **Quality** | Removed global `pd.set_option` from module scope | `exchange_agent.py` | `f1a296a` |
| 4.3 | **Quality** | `queue.PriorityQueue` → `heapq` (removed mutex overhead) | `kernel.py` | `f1a296a` |

**Skipped (low impact):** Subscription data structure O(N) cancel lookup (mitigated by `break` fix). Message/Order ID counters already use `itertools.count()` (GIL-safe). Oracle RNG verified safe (injected `RandomState`).

**Tests added:** `abides-core/tests/test_kernel.py`, updated `abides-markets/tests/orderbook/test_data_methods.py`.

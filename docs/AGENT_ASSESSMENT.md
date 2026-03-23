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
| `type(self) is NoiseAgent` guard | Code smell | Line 135 — fragile isinstance-vs-type check prevents subclass reuse. |

### 2.3 ValueAgent

**Role**: Bayesian fundamental-value estimator. Core of price discovery.

| Concern | Severity | Detail |
|---------|:--------:|--------|
| **Type annotation bug** | Low | Line 29: `log_orders: float = False` — should be `bool`. |
| `cancel_all_orders()` then immediately re-queries | Medium | Lines 132-138: on every wakeup the agent cancels all open orders and re-places them. This generates 2x message traffic (cancel + new order) instead of amending. With many ValueAgents this multiplies exchange load. |
| `buy` variable conflation | Low | Line 245: `buy` is set to `True`/`False` in the `if/elif` branches but to `self.random_state.randint(0, 1 + 1)` (integer 0 or 1) in the `else`. Then `Side.BID if buy == 1` — works but is fragile mixed-type logic. |
| `type(self) is ValueAgent` guard | Code smell | Same pattern as NoiseAgent. |
| Hard-coded `depth_spread = 2` | Low | Not configurable via constructor or config system. |

### 2.4 MomentumAgent

**Role**: Moving-average crossover (20 vs 50 bar).

| Concern | Severity | Detail |
|---------|:--------:|--------|
| **Unsafe `known_bids` access in subscribe mode** | High | Line 106: `self.known_bids[self.symbol]` — will raise `KeyError` if the first message arrives before any spread data. Should use `.get(symbol, [])`. |
| Unbounded `mid_list` growth | Medium | Line 112: `self.mid_list.append(...)` grows forever. In a 6.5-hour trading day at 1s frequency = 23,400 entries. `ma()` recomputes `np.cumsum` over the full array each time — O(n) growing cost. Should use a rolling window / `deque`. |
| Hard-coded MA windows (20, 50) | Design | Not configurable. A production momentum agent needs tunable fast/slow windows. |
| No position management | Design | Keeps buying/selling without any position limit or reversal logic. Can accumulate unbounded inventory. |

### 2.5 AdaptiveMarketMakerAgent

**Role**: Ladder market-maker with inventory skew and adaptive spread.

| Concern | Severity | Detail |
|---------|:--------:|--------|
| **Bug: wrong state key in subscribe mode** | Critical | Line 326: `self.state["MARKET_DATA"]` should be `self.state["AWAITING_MARKET_DATA"]`. This is a `KeyError` that will crash the agent whenever it runs in subscription mode. |
| Cancel-then-repost strategy | Medium | Every wakeup calls `cancel_all_orders()` then places a fresh ladder. This is 2 × `num_ticks` cancellation messages + 2 × `num_ticks` new orders. Amend-in-place would halve message volume. |
| No P&L / risk tracking | Design | No max-position limit, no loss threshold, no end-of-day flatten. A production MM needs all of these. |
| `subscribe_freq` as `float` | Low | Declared as `float` but used as `int` in ns comparisons. |

### 2.6 POVExecutionAgent

**Role**: Percentage-of-volume execution algorithm.

| Concern | Severity | Detail |
|---------|:--------:|--------|
| `last_bid`/`last_ask` typed as `Optional[float]` | Low | Lines 132-133: prices should be `Optional[int]` (integer cents). |
| Not tracking execution fill prices | Medium | `order_executed` tracks quantity but not price. Cannot compute VWAP of executed fills for slippage analysis. |
| Market orders only | Design | Uses exclusively market orders for fills. A production POV agent should also support limit orders with active crossing for better price improvement. |
| No urgency parameter | Design | Real POV algos have urgency controls (e.g., ramp rate near close). |

### 2.7 TradingAgent Base (1,250 LOC)

| Concern | Severity | Detail |
|---------|:--------:|--------|
| `deepcopy` on every order placement | Medium | Lines 556, 611, 647: every `place_limit_order`/`place_market_order` deep-copies the order. Performance cost scales linearly with order flow. |
| `mark_to_market` KeyError risk | Medium | Line 1166: `self.last_trade[symbol]` will raise `KeyError` if no trade has occurred for a held symbol. |
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

3. **Unsafe dictionary access patterns**: Multiple agents access `self.known_bids[symbol]` directly instead of `.get(symbol, [])`, risking `KeyError` at runtime.

---

## 4. Recommendations

### 4.1 Immediate Fixes (Bugs)

| Fix | File | Line |
|-----|------|------|
| Change `self.state["MARKET_DATA"]` → `self.state["AWAITING_MARKET_DATA"]` | `adaptive_market_maker_agent.py` | 326 |
| Change `log_orders: float` → `log_orders: bool` | `value_agent.py` | 29 |
| Use `.get(symbol, [])` in MomentumAgent subscribe mode | `momentum_agent.py` | 106 |

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

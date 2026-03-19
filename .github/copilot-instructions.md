# ABIDES — Copilot Instructions

ABIDES is a **discrete-event market simulation**. Agents react to events — there are no loops.

## Critical rules

### Prices are integer cents everywhere
`$100.00 = 10_000`. Never use floats for prices. Display only: `f"${price / 100:.2f}"`.

### Market data is None/empty until received asynchronously

| Field | Empty state |
|-------|-------------|
| `self.mkt_open`, `self.mkt_close` | `None` until `MarketHoursMsg` received |
| `self.known_bids.get(symbol, [])` | `KeyError` before first spread response; `[]` if book side empty |
| `self.known_asks.get(symbol, [])` | same |
| `self.last_trade.get(symbol)` | `KeyError` before first trade |
| `L1DataMsg.bid / .ask` | `None` when book side empty |
| `L2DataMsg.bids / .asks` | `[]` when book side empty |

Always guard: `bids = self.known_bids.get(symbol, []); bid = bids[0][0] if bids else None`

### Nothing is synchronous
`get_current_spread()` sends a message. The response arrives later in `receive_message()`.
The only agent entry points are `wakeup()` and `receive_message()`.

### Check `super().wakeup()` return value
`TradingAgent.wakeup()` returns `False` when market hours are unknown or market is closed.
Always: `if not super().wakeup(current_time): return`

## Custom agent pattern
Implement `TradingStrategy` → wrap in `AbidesStrategyAdapter(TradingAgent)`.
See: `docs/ABIDES_CUSTOM_AGENT_IMPLEMENTATION_GUIDE.md`

## External data / oracle injection
Use `ExternalDataOracle` with `BatchDataProvider` or `PointDataProvider`.
See: `abides-markets/abides_markets/oracles/`

## Full reference
- `docs/ABIDES_LLM_INTEGRATION_GOTCHAS.md` — all None/NaN traps, safe patterns
- `docs/ABIDES_CUSTOM_AGENT_IMPLEMENTATION_GUIDE.md` — full adapter pattern
- `docs/ABIDES_DATA_EXTRACTION.md` — parsing logs (`parse_logs_df`) and L1/L2 book history
- `docs/PARALLEL_SIMULATION_GUIDE.md` — multiprocessing, RNG hierarchy, log layout
- `docs/ABIDES_REMEDIATION_PLAN.md` — changelog of performance/correctness fixes

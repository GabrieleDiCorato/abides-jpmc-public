#!/usr/bin/env python
"""Hands-on evaluation of all ABIDES agents and features.

Run as: uv run python version_testing/evaluate_all_agents.py
"""

from __future__ import annotations

import traceback

# ===========================================================================
# 1. Verify imports
# ===========================================================================
print("=" * 70)
print("ABIDES — Full Feature Evaluation")
print("=" * 70)

from abides_markets.config_system import (
    SimulationBuilder,
    list_agent_types,
    list_templates,
)
from abides_markets.simulation import ResultProfile, run_simulation

print("\n[1] Templates available:")
for t in list_templates():
    tag = " (overlay)" if t["is_overlay"] else ""
    print(f"    {t['name']}{tag}: {t['description']}")

print("\n[2] Agent types registered:")
for a in list_agent_types():
    print(f"    {a['name']:<25} category={a['category']:<15}")

# ===========================================================================
# 2. Basic rmsc04 run — smoke test
# ===========================================================================
print("\n" + "=" * 70)
print("TEST A: Basic rmsc04 simulation (SUMMARY profile)")
print("=" * 70)

config_a = SimulationBuilder().from_template("rmsc04").seed(42).build()
result_a = run_simulation(config_a, profile=ResultProfile.SUMMARY)
print(f"  Wall clock: {result_a.metadata.wall_clock_elapsed_s:.2f}s")
print(f"  Tickers: {result_a.metadata.tickers}")
for sym, mkt in result_a.markets.items():
    liq = mkt.liquidity
    last = f"${liq.last_trade_cents / 100:.2f}" if liq.last_trade_cents else "None"
    vwap = f"${liq.vwap_cents / 100:.2f}" if liq.vwap_cents else "None"
    print(
        f"  {sym}: last={last}  VWAP={vwap}  "
        f"vol={liq.total_exchanged_volume:,}  "
        f"no_bid={liq.pct_time_no_bid:.1f}%  no_ask={liq.pct_time_no_ask:.1f}%"
    )
print(f"  Agents: {len(result_a.agents)}")

# PnL distribution
pnls = [a.pnl_cents for a in result_a.agents]
print(f"  PnL range: ${min(pnls)/100:.2f} to ${max(pnls)/100:.2f}")
print(f"  Avg PnL: ${sum(pnls)/len(pnls)/100:.2f}")

# ===========================================================================
# 3. QUANT profile — full time-series + trade attribution + equity curves
# ===========================================================================
print("\n" + "=" * 70)
print("TEST B: QUANT profile with all data extraction")
print("=" * 70)

config_b = SimulationBuilder().from_template("rmsc04").seed(123).build()
result_b = run_simulation(config_b, profile=ResultProfile.QUANT)

sym = result_b.metadata.tickers[0]
mkt_b = result_b.markets[sym]

# L1 series
if mkt_b.l1_series:
    l1 = mkt_b.l1_series
    print(f"  L1 series: {len(l1.times_ns)} snapshots")
    # Check for None in bid/ask prices
    non_none_bids = sum(1 for p in l1.bid_prices if p is not None)
    non_none_asks = sum(1 for p in l1.ask_prices if p is not None)
    print(f"    Non-None bids: {non_none_bids}/{len(l1.bid_prices)}")
    print(f"    Non-None asks: {non_none_asks}/{len(l1.ask_prices)}")
else:
    print("  L1 series: NOT POPULATED (unexpected!)")

# L2 series
if mkt_b.l2_series:
    l2 = mkt_b.l2_series
    print(f"  L2 series: {len(l2.times_ns)} snapshots")
else:
    print("  L2 series: NOT POPULATED (unexpected!)")

# Trade attribution
if mkt_b.trades:
    print(f"  Trade attribution: {len(mkt_b.trades)} executions")
    t0 = mkt_b.trades[0]
    print(
        f"    First trade: passive={t0.passive_agent_id} aggressor={t0.aggressive_agent_id} "
        f"side={t0.side} price=${t0.price_cents/100:.2f} qty={t0.quantity}"
    )
else:
    print("  Trade attribution: NOT POPULATED (unexpected!)")

# Equity curves
agents_with_curves = [a for a in result_b.agents if a.equity_curve is not None]
print(f"  Agents with equity curves: {len(agents_with_curves)}/{len(result_b.agents)}")
if agents_with_curves:
    # Show top 3 by max drawdown
    by_dd = sorted(
        agents_with_curves,
        key=lambda a: a.equity_curve.max_drawdown_cents,
        reverse=True,
    )
    print("  Top 3 drawdowns:")
    for a in by_dd[:3]:
        ec = a.equity_curve
        print(
            f"    [{a.agent_id}] {a.agent_type}: max_dd=${ec.max_drawdown_cents/100:.2f}  "
            f"fills={len(ec.times_ns)}"
        )

# ===========================================================================
# 4. TWAP execution agent
# ===========================================================================
print("\n" + "=" * 70)
print("TEST C: TWAP execution agent")
print("=" * 70)

try:
    config_c = (
        SimulationBuilder()
        .from_template("rmsc04")
        .enable_agent(
            "twap_execution", count=1, direction="BID", quantity=500, freq="30s"
        )
        .seed(42)
        .build()
    )
    result_c = run_simulation(config_c, profile=ResultProfile.QUANT)

    exec_agents_c = [a for a in result_c.agents if a.execution_metrics is not None]
    for ea in exec_agents_c:
        em = ea.execution_metrics
        print(f"  [{ea.agent_id}] {ea.agent_type}")
        print(
            f"    Target: {em.target_quantity}  Filled: {em.filled_quantity}  Rate: {em.fill_rate_pct:.1f}%"
        )
        if em.avg_fill_price_cents:
            print(f"    Avg fill: ${em.avg_fill_price_cents/100:.2f}")
        if em.vwap_cents:
            print(f"    Session VWAP: ${em.vwap_cents/100:.2f}")
        if em.vwap_slippage_bps is not None:
            print(f"    VWAP slippage: {em.vwap_slippage_bps} bps")
        if em.arrival_price_cents:
            print(f"    Arrival price: ${em.arrival_price_cents/100:.2f}")
        if em.implementation_shortfall_bps is not None:
            print(f"    Impl shortfall: {em.implementation_shortfall_bps} bps")
        # Check equity curve
        if ea.equity_curve:
            print(
                f"    Equity curve: {len(ea.equity_curve.times_ns)} fills  "
                f"max_dd=${ea.equity_curve.max_drawdown_cents/100:.2f}"
            )
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()

# ===========================================================================
# 5. VWAP execution agent
# ===========================================================================
print("\n" + "=" * 70)
print("TEST D: VWAP execution agent (with custom volume profile)")
print("=" * 70)

try:
    config_d = (
        SimulationBuilder()
        .from_template("rmsc04")
        .enable_agent(
            "vwap_execution",
            count=1,
            direction="ASK",
            quantity=500,
            freq="30s",
            volume_profile=[0.3, 0.1, 0.1, 0.1, 0.1, 0.3],
        )
        .seed(42)
        .build()
    )
    result_d = run_simulation(config_d, profile=ResultProfile.QUANT)

    exec_agents_d = [a for a in result_d.agents if a.execution_metrics is not None]
    for ea in exec_agents_d:
        em = ea.execution_metrics
        print(f"  [{ea.agent_id}] {ea.agent_type}")
        print(
            f"    Target: {em.target_quantity}  Filled: {em.filled_quantity}  Rate: {em.fill_rate_pct:.1f}%"
        )
        if em.avg_fill_price_cents:
            print(f"    Avg fill: ${em.avg_fill_price_cents/100:.2f}")
        if em.vwap_slippage_bps is not None:
            print(f"    VWAP slippage: {em.vwap_slippage_bps} bps")
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()

# ===========================================================================
# 6. Mean reversion agent
# ===========================================================================
print("\n" + "=" * 70)
print("TEST E: Mean reversion agent")
print("=" * 70)

try:
    config_e = (
        SimulationBuilder()
        .from_template("rmsc04")
        .enable_agent("mean_reversion", count=5)
        .seed(42)
        .build()
    )
    result_e = run_simulation(config_e, profile=ResultProfile.SUMMARY)

    mr_agents = [a for a in result_e.agents if a.agent_type == "MeanReversionAgent"]
    print(f"  Mean reversion agents: {len(mr_agents)}")
    for a in mr_agents:
        print(f"    [{a.agent_id}] PnL: ${a.pnl_cents/100:.2f} ({a.pnl_pct:+.2f}%)")
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()

# ===========================================================================
# 7. AMM flatten — verify it works in a simulation
# ===========================================================================
print("\n" + "=" * 70)
print("TEST F: AMM with end-of-day flatten")
print("=" * 70)

try:
    config_f = (
        SimulationBuilder()
        .from_template("rmsc04")
        .enable_agent("adaptive_market_maker", count=1, flatten_before_close="5min")
        .seed(42)
        .build()
    )
    result_f = run_simulation(config_f, profile=ResultProfile.FULL)

    mm_agents = [a for a in result_f.agents if "MarketMaker" in a.agent_type]
    print(f"  Market makers: {len(mm_agents)}")
    for a in mm_agents:
        sym = [s for s in a.final_holdings if s != "CASH"]
        pos = {s: a.final_holdings[s] for s in sym}
        print(
            f"    [{a.agent_id}] {a.agent_type}: PnL=${a.pnl_cents/100:.2f}  "
            f"positions={pos}"
        )

    # Check raw logs for AMM_FLATTEN event
    if result_f.logs is not None:
        flatten_events = result_f.logs[result_f.logs["EventType"] == "AMM_FLATTEN"]
        print(f"  AMM_FLATTEN events in logs: {len(flatten_events)}")
    else:
        print("  (logs not available)")
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()

# ===========================================================================
# 8. Stop orders — regression check (stop orders are agent-level, so just
#    verify the exchange can handle StopOrderMsg in a simulation context)
# ===========================================================================
print("\n" + "=" * 70)
print("TEST G: Stop order integration (via low-level compile)")
print("=" * 70)
print("  (Stop orders tested via unit tests — no built-in agent uses them yet)")

# ===========================================================================
# 9. Thin market stress test
# ===========================================================================
print("\n" + "=" * 70)
print("TEST H: Thin market template (stress test — low liquidity)")
print("=" * 70)

try:
    config_h = SimulationBuilder().from_template("thin_market").seed(42).build()
    result_h = run_simulation(config_h, profile=ResultProfile.QUANT)

    sym = result_h.metadata.tickers[0]
    liq = result_h.markets[sym].liquidity
    last = f"${liq.last_trade_cents / 100:.2f}" if liq.last_trade_cents else "None"
    vwap = f"${liq.vwap_cents / 100:.2f}" if liq.vwap_cents else "None"
    print(f"  {sym}: last={last}  VWAP={vwap}  " f"vol={liq.total_exchanged_volume:,}")
    print(f"  No-bid: {liq.pct_time_no_bid:.1f}%  No-ask: {liq.pct_time_no_ask:.1f}%")
    print(
        f"  Agents: {len(result_h.agents)}  "
        f"Trade attributions: {len(result_h.markets[sym].trades or [])}"
    )
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()

# ===========================================================================
# 10. Liquid market with execution overlays
# ===========================================================================
print("\n" + "=" * 70)
print("TEST I: Liquid market + TWAP + VWAP + POV competing")
print("=" * 70)

try:
    config_i = (
        SimulationBuilder()
        .from_template("liquid_market")
        .enable_agent(
            "twap_execution", count=1, direction="BID", quantity=1000, freq="1min"
        )
        .enable_agent(
            "vwap_execution", count=1, direction="BID", quantity=1000, freq="1min"
        )
        .enable_agent("pov_execution", count=1, direction="ASK", quantity=1000, pov=0.1)
        .seed(42)
        .build()
    )
    result_i = run_simulation(config_i, profile=ResultProfile.QUANT)

    print("  Execution agent comparison:")
    exec_agents_i = [a for a in result_i.agents if a.execution_metrics is not None]
    for ea in sorted(exec_agents_i, key=lambda a: a.agent_type):
        em = ea.execution_metrics
        avg = (
            f"${em.avg_fill_price_cents/100:.2f}" if em.avg_fill_price_cents else "N/A"
        )
        vwap_slip = (
            f"{em.vwap_slippage_bps}bps" if em.vwap_slippage_bps is not None else "N/A"
        )
        is_bps = (
            f"{em.implementation_shortfall_bps}bps"
            if em.implementation_shortfall_bps is not None
            else "N/A"
        )
        print(
            f"    {ea.agent_type:<25} fill={em.fill_rate_pct:5.1f}%  "
            f"avg={avg}  "
            f"vwap_slip={vwap_slip:>6}  IS={is_bps:>6}"
        )
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()

# ===========================================================================
# 11. Summary via narrative
# ===========================================================================
print("\n" + "=" * 70)
print("TEST J: result.summary() narrative")
print("=" * 70)

try:
    print(result_a.summary())
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()

# ===========================================================================
# Final
# ===========================================================================
print("\n" + "=" * 70)
print("ALL TESTS COMPLETE")
print("=" * 70)

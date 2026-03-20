"""Compiler: converts a SimulationConfig into the runtime dict for Kernel.

The output dict has the exact same shape as what ``rmsc04.build_config()``
returns, so it works with ``abides.run()``, gymnasium envs, and
``config_add_agents()`` without any changes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from abides_core import NanosecondTime
from abides_core.utils import str_to_ns

from abides_markets.agents import ExchangeAgent
from abides_markets.config_system.agent_configs import AgentCreationContext
from abides_markets.config_system.models import (
    ExternalDataOracleConfig,
    MeanRevertingOracleConfig,
    SimulationConfig,
    SparseMeanRevertingOracleConfig,
)
from abides_markets.config_system.registry import registry
from abides_markets.utils import generate_latency_model


def compile(config: SimulationConfig) -> dict[str, Any]:
    """Compile a declarative SimulationConfig into a Kernel-compatible runtime dict.

    The output dict matches the format returned by ``rmsc04.build_config()``::

        {
            "seed": int,
            "start_time": NanosecondTime,
            "stop_time": NanosecondTime,
            "agents": List[Agent],
            "agent_latency_model": LatencyModel,
            "default_computation_delay": int,
            "custom_properties": {"oracle": Oracle},
            "random_state_kernel": np.random.RandomState,
            "stdout_log_level": str,
        }
    """
    # ── Resolve seed ──────────────────────────────────────────────
    if config.simulation.seed == "random":
        seed = int(datetime.now().timestamp() * 1_000_000) % (2**32 - 1)
    else:
        seed = config.simulation.seed

    master_rng = np.random.RandomState(seed)

    # ── Timestamps ────────────────────────────────────────────────
    date_ns: NanosecondTime = pd.to_datetime(config.market.date).value
    mkt_open: NanosecondTime = date_ns + str_to_ns(config.market.start_time)
    mkt_close: NanosecondTime = date_ns + str_to_ns(config.market.end_time)

    # ── Oracle ────────────────────────────────────────────────────
    oracle_rng = np.random.RandomState(
        seed=master_rng.randint(low=0, high=2**32, dtype="uint64")
    )
    oracle = _build_oracle(config, mkt_open, mkt_close, oracle_rng)

    # ── Create shared context for agent factories ─────────────────
    oracle_r_bar = _get_oracle_r_bar(config)
    context = AgentCreationContext(
        ticker=config.market.ticker,
        mkt_open=mkt_open,
        mkt_close=mkt_close,
        log_orders=config.simulation.log_orders,
        oracle_r_bar=oracle_r_bar,
        date_ns=date_ns,
    )

    # ── Agents ────────────────────────────────────────────────────
    agents = []
    agent_count = 0
    per_agent_computation_delays: dict[int, int] = {}

    # Exchange is always agent id=0
    exc = config.market.exchange
    agents.append(
        ExchangeAgent(
            id=0,
            name="EXCHANGE_AGENT",
            type="ExchangeAgent",
            mkt_open=mkt_open,
            mkt_close=mkt_close,
            symbols=[config.market.ticker],
            book_logging=exc.book_logging,
            book_log_depth=exc.book_log_depth,
            log_orders=exc.log_orders,
            pipeline_delay=exc.pipeline_delay,
            computation_delay=exc.computation_delay,
            stream_history=exc.stream_history_length,
            random_state=np.random.RandomState(
                seed=master_rng.randint(low=0, high=2**32, dtype="uint64")
            ),
        )
    )
    agent_count += 1

    # Instantiate each enabled agent group via the registry
    for agent_type_name, group in config.agents.items():
        if not group.enabled or group.count == 0:
            continue

        entry = registry.get(agent_type_name)
        # Validate params against the registered config model
        agent_config = entry.config_model(**group.params)

        new_agents = agent_config.create_agents(
            count=group.count,
            id_start=agent_count,
            master_rng=master_rng,
            context=context,
        )
        agents.extend(new_agents)

        # Record per-agent computation delay overrides
        if agent_config.computation_delay is not None:
            for agent_id in range(agent_count, agent_count + group.count):
                per_agent_computation_delays[agent_id] = agent_config.computation_delay

        agent_count += group.count

    # ── Kernel seed ───────────────────────────────────────────────
    random_state_kernel = np.random.RandomState(
        seed=master_rng.randint(low=0, high=2**32, dtype="uint64")
    )

    # ── Latency ───────────────────────────────────────────────────
    latency_rng = np.random.RandomState(
        seed=master_rng.randint(low=0, high=2**32, dtype="uint64")
    )
    latency_model = generate_latency_model(
        agent_count,
        latency_rng,
        latency_type=config.infrastructure.latency.type,
    )

    # ── Assemble runtime dict ─────────────────────────────────────
    kernel_start = date_ns
    kernel_stop = mkt_close + str_to_ns("1s")

    runtime: dict[str, Any] = {
        "seed": seed,
        "start_time": kernel_start,
        "stop_time": kernel_stop,
        "agents": agents,
        "agent_latency_model": latency_model,
        "default_computation_delay": config.infrastructure.default_computation_delay,
        "custom_properties": {"oracle": oracle},
        "random_state_kernel": random_state_kernel,
        "stdout_log_level": config.simulation.log_level,
    }

    if per_agent_computation_delays:
        runtime["per_agent_computation_delays"] = per_agent_computation_delays

    return runtime


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_oracle(config, mkt_open, mkt_close, oracle_rng):
    """Construct the oracle from the config's oracle section."""
    oc = config.market.oracle

    if isinstance(oc, SparseMeanRevertingOracleConfig):
        from abides_markets.oracles import SparseMeanRevertingOracle

        # Pass market close as the end of the noise agent window for consistency with rmsc04
        # rmsc04 uses NOISE_MKT_CLOSE = date + "16:00:00"
        date_ns = pd.to_datetime(config.market.date).value
        noise_mkt_close = date_ns + str_to_ns("16:00:00")

        symbols = {
            config.market.ticker: {
                "r_bar": oc.r_bar,
                "kappa": oc.kappa,
                "sigma_s": oc.sigma_s,
                "fund_vol": oc.fund_vol,
                "megashock_lambda_a": oc.megashock_lambda_a,
                "megashock_mean": oc.megashock_mean,
                "megashock_var": oc.megashock_var,
            }
        }
        return SparseMeanRevertingOracle(mkt_open, noise_mkt_close, symbols, oracle_rng)

    elif isinstance(oc, MeanRevertingOracleConfig):
        from abides_markets.oracles import MeanRevertingOracle

        symbols = {
            config.market.ticker: {
                "r_bar": oc.r_bar,
                "kappa": oc.kappa,
                "sigma_s": oc.sigma_s,
            }
        }
        return MeanRevertingOracle(mkt_open, mkt_close, symbols, oracle_rng)

    elif isinstance(oc, ExternalDataOracleConfig):

        raise NotImplementedError(
            "ExternalDataOracle loading from data_path is not yet implemented. "
            "Use compile() with a pre-built oracle injected via custom_properties."
        )
    else:
        raise ValueError(f"Unknown oracle type: {type(oc)}")


def _get_oracle_r_bar(config) -> int:
    """Extract r_bar from oracle config for derived parameters."""
    oc = config.market.oracle
    if isinstance(oc, (SparseMeanRevertingOracleConfig, MeanRevertingOracleConfig)):
        return oc.r_bar
    return 100_000  # default fallback
